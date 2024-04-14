import math

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


class BadDiffusion(GaussianDiffusion):
    def __init__(self, model, image_size, timesteps, sampling_timesteps, objective, trigger):
        super().__init__(model, image_size=image_size, timesteps=timesteps, sampling_timesteps=sampling_timesteps,
                         objective=objective)
        self.trigger = trigger

    def bad_p_losses(self, x_start, t, mode, noise=None, offset_noise_strength=None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            # target = x_start
            if mode == 1:
                target = self.trigger
            else:
                target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, mode, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.bad_p_losses(img, t, mode, *args, **kwargs)


class BadTrainer(denoising_diffusion_pytorch.Trainer):
    def __init__(self, diffusion, good_folder, train_batch_size, train_lr, train_num_steps,
                 gradient_accumulate_every, ema_decay, amp, calculate_fid, bad_folder=None):
        super().__init__(diffusion_model=diffusion, folder=good_folder, train_batch_size=train_batch_size,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         gradient_accumulate_every=gradient_accumulate_every, ema_decay=ema_decay, amp=amp,
                         calculate_fid=calculate_fid)
        if bad_folder is not None:
            self.bad_ds = Dataset(bad_folder, self.image_size, augment_horizontal_flip=True, convert_image_to='RGB')
            bad_dl = DataLoader(self.bad_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                num_workers=4)
            bad_dl = self.accelerator.prepare(bad_dl)
            self.bad_dl = cycle(bad_dl)
        # print('bad trainer')

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for mode in range(self.gradient_accumulate_every):
                    if mode == 0:
                        data = next(self.dl).to(device)
                    elif mode == 1:
                        data = next(self.bad_dl).to(device)

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


if __name__ == '__main__':
    triger_path = '/home/chengyiqiu/code/Diffusion-Backdoor-Embed/resource/badnet/trigger_image_grid.png'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32))
    ])
    triger = Image.open(triger_path)
    triger = transform(triger).to('cuda:0')
    print(triger.shape)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    diffusion = BadDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        objective='pred_x0',
        trigger=triger
    )

    trainer = BadTrainer(
        diffusion,
        bad_folder='../dataset/dataset-cifar10-badnet-trigger_image_grid',
        good_folder='../dataset/dataset-cifar10-good',
        train_batch_size=128,
        train_lr=8e-5,
        # train_num_steps=700000,  # total training steps
        train_num_steps=1000,
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True  # whether to calculate fid during training
    )

    trainer.train()
