import argparse
import math
import ast
import PIL.Image
import denoising_diffusion_pytorch
import torch
import torchvision.transforms
from torchvision import utils
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch.nn.functional as F
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, rearrange, random, reduce, extract, cycle, \
    Dataset, divisible_by, num_to_groups
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import sys

sys.path.append('../')
from tools import tg_bot


class BadDiffusion(GaussianDiffusion):
    @property
    def device(self):
        return self._device

    def __init__(self, model, image_size, timesteps, sampling_timesteps, objective, trigger, loss_mode,
                 factor_list, device):
        super().__init__(model, image_size=image_size, timesteps=timesteps, sampling_timesteps=sampling_timesteps,
                         objective=objective)
        self.trigger = trigger
        self.loss_mode = loss_mode
        self.factor_list = factor_list
        self.device = device

    def bad_p_losses(self, x_start, t, mode, noise=None, offset_noise_strength=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        model_out = self.model(x, t, x_self_cond)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        if mode == 0:  # benign data loss
            loss = F.mse_loss(model_out, target, reduction='none')
            loss = reduce(loss, 'b ... -> b', 'mean')
            loss = loss * extract(self.loss_weight, t, loss.shape)
            loss = loss.mean()
            i = 0
        else:  # trigger data
            # use SSIM and MSE
            import sys
            sys.path.append('..')
            from tools import diffusion_loss
            mask = PIL.Image.open('../resource/badnet/trigger_image.png')
            trans = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), torchvision.transforms.Resize((32, 32))
            ])
            mask = trans(mask).to(self.device)
            p_trigger = mask * model_out
            x_p_no_trigger = (1 - mask) * model_out
            x_no_trigger = (1 - mask) * target
            loss_fn = diffusion_loss.loss_dict.get(self.loss_mode)
            loss = loss_fn(p_trigger, self.trigger, x_p_no_trigger, x_no_trigger, self.factor_list)
        return loss

    def forward(self, img, mode, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.bad_p_losses(img, t, mode, *args, **kwargs)

    @device.setter
    def device(self, value):
        self._device = value


class BadTrainer(denoising_diffusion_pytorch.Trainer):
    def __init__(self, diffusion, good_folder, train_batch_size, train_lr, train_num_steps,
                 gradient_accumulate_every, ratio, results_folder, ema_decay, amp, calculate_fid, bad_folder=None):
        super().__init__(diffusion_model=diffusion, folder=good_folder, train_batch_size=train_batch_size,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         gradient_accumulate_every=gradient_accumulate_every, ema_decay=ema_decay, amp=amp,
                         calculate_fid=calculate_fid)
        self.ratio = ratio
        self.results_folder = ""
        if bad_folder is not None:
            self.bad_ds = Dataset(bad_folder, self.image_size, augment_horizontal_flip=True, convert_image_to='RGB')
            bad_dl = DataLoader(self.bad_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                num_workers=4)
            bad_dl = self.accelerator.prepare(bad_dl)
            self.bad_dl = cycle(bad_dl)
        # print('bad trainer')

    def train(self):
        accelerator = self.accelerator
        device = self.device
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for mode in range(self.gradient_accumulate_every):
                    # if mode == 0:
                    #     data = next(self.dl).to(device)
                    # elif mode == 1:
                    #     import random
                    #     rand_num = random.random()
                    #     if rand_num < 0.8:
                    #         data = next(self.dl).to(device)
                    #         mode = 0
                    #     else:
                    #         data = next(self.bad_dl).to(device)
                    #         mode = 1
                    import random
                    if random.random() < self.ratio:
                        data = next(self.bad_dl).to(device)
                        mode = 1
                    else:
                        data = next(self.dl).to(device)
                        mode = 0
                    with self.accelerator.autocast():
                        loss = self.model(data, mode)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                         nrow=int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


def get_args():
    parser = argparse.ArgumentParser(description='This script does amazing things.')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for processing')
    parser.add_argument('--step', type=int, default=10000, help='Number of steps for the diffusion model')
    parser.add_argument('--loss_mode', type=int, default=4, help='Mode for loss function')
    parser.add_argument('--factor', type=str, default='[1, 1, 1]',
                        help='Factor to be used in the loss function, given as a string representation of a list')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the process on (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--ratio', type=float, default=0.1, help='A poisoning ratio value to be used in calculations')
    parser.add_argument('--results_folder', type=str, default='./results', help='Folder to save results')
    parser.add_help = True
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    batch = args.batch
    train_num_steps = args.step
    loss_mode = args.loss_mode
    device = args.device
    ratio = args.ratio
    results_folder = args.results_folder
    factor_list = ast.literal_eval(args.factor)
    triger_path = '../resource/badnet/trigger_image_grid.png'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32))
    ])
    triger = Image.open(triger_path)
    triger = transform(triger).to(device)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    model = model.to(device)
    diffusion = BadDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        objective='pred_x0',
        trigger=triger,
        loss_mode=loss_mode,
        factor_list=factor_list,
        device=device,
    )

    trainer = BadTrainer(
        diffusion,
        bad_folder='../dataset/dataset-cifar10-badnet-trigger_image_grid',
        good_folder='../dataset/dataset-cifar10-good',
        train_batch_size=batch,
        train_lr=8e-5,
        # train_num_steps=700000,  # total training steps
        train_num_steps=train_num_steps,
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True,  # whether to calculate fid during training
        ratio=ratio,
        results_folder=results_folder,
    )

    trainer.train()
    tg_bot.send2bot('pc train diffusion down', 'diffusion')
