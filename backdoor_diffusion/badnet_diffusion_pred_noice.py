import argparse
import math
import ast
import time

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
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('../')
from tools import img
from tools import tg_bot
from tools import diffusion_loss
from tools.time import sleep_cat
from tools.img import cal_ssim, cal_ppd


class BadDiffusion(GaussianDiffusion):
    @property
    def device(self):
        return self._device

    def __init__(self, model, image_size, timesteps, sampling_timesteps, objective, trigger,
                 factor_list, device):
        super().__init__(model, image_size=image_size, timesteps=timesteps, sampling_timesteps=sampling_timesteps,
                         objective=objective)
        self.trigger = trigger
        self.factor_list = factor_list
        self.device = device

    def train_mode_p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

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
        else:  # trigger data
            import sys
            sys.path.append('..')
            mask = PIL.Image.open('../resource/badnet/trigger_image.png')
            trans = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), torchvision.transforms.Resize((32, 32))
            ])
            mask = trans(mask).to(self.device)
            loss = F.mse_loss(model_out * (1 - mask) + mask * (1 - self.trigger), target, reduction='none')
            loss = reduce(loss, 'b ... -> b', 'mean')
            loss = loss * extract(self.loss_weight, t, loss.shape)
            loss = loss.mean()
            # print(loss)
        return loss

    def forward(self, img, mode, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.bad_p_losses(img, t, mode, *args, **kwargs)

    def train_mode_ddim_sample(self, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True,
                                                             rederive_pred_noise=True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @device.setter
    def device(self, value):
        self._device = value


class BadTrainer(denoising_diffusion_pytorch.Trainer):
    def __init__(self, diffusion, good_folder, train_batch_size, train_lr, train_num_steps,
                 gradient_accumulate_every, ratio, results_folder, server, save_and_sample_every, ema_decay, amp,
                 calculate_fid,
                 bad_folder=None):
        super().__init__(diffusion_model=diffusion, folder=good_folder, train_batch_size=train_batch_size,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         gradient_accumulate_every=gradient_accumulate_every, ema_decay=ema_decay, amp=amp,
                         calculate_fid=calculate_fid, results_folder=results_folder,
                         save_and_sample_every=save_and_sample_every)
        self.ratio = ratio
        self.server = server
        if bad_folder is not None:
            self.bad_ds = Dataset(bad_folder, self.image_size, augment_horizontal_flip=True, convert_image_to='RGB')
            bad_dl = DataLoader(self.bad_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                num_workers=4)
            bad_dl = self.accelerator.prepare(bad_dl)
            self.bad_dl = cycle(bad_dl)

    def train(self):
        accelerator = self.accelerator
        device = self.device
        loss_list = []
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                if self.server == 'lab':
                    sleep_cat()
                total_loss = 0.
                for mode in range(self.gradient_accumulate_every):
                    import random
                    if random.random() < self.ratio:
                        data = next(self.bad_dl).to(device)
                        mode = 1  # poisoning
                    else:
                        data = next(self.dl).to(device)
                        mode = 0
                    with self.accelerator.autocast():
                        loss = self.model(data, mode)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                formatted_loss = format(total_loss, '.4f')
                loss_list.append({
                    'loss': float(formatted_loss),
                    'mode': mode
                })
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
        # torch.save(loss_list, str(self.results_folder / f'metrics.pth'))
        accelerator.print('training complete')
        return loss_list


def get_args():
    parser = argparse.ArgumentParser(description='This script does amazing things.')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for processing')
    parser.add_argument('--step', type=int, default=10000, help='Number of steps for the diffusion model')
    parser.add_argument('--loss_mode', type=int, default=4, help='Mode for loss function')
    parser.add_argument('--factor', type=str, default='[1, 1, 1]',
                        help='Factor to be used in the loss function, given as a string representation of a list')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the process on (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--ratio', type=float, default=0.1, help='A poisoning ratio value to be used in calculations')
    parser.add_argument('--results_folder', type=str, default='./res_test', help='Folder to save results')
    parser.add_argument('--server', type=str, default='pc', help='which server you use, lab, pc, or lv')
    parser.add_argument('--save_and_sample_every', type=int, default=0, help='save every step')
    parser.add_help = True
    return parser.parse_args()


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg: DictConfig):
    print(cfg)
    unet_cfg = cfg.noise_predictor
    diff_cfg = cfg.diffusion
    trainer_cfg = cfg.trainer
    device = diff_cfg.device
    import os
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    triger_path = diff_cfg.trigger
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32))
    ])
    triger = Image.open(triger_path)
    triger = transform(triger).to(device)
    model = Unet(
        dim=unet_cfg.dim,
        dim_mults=tuple(map(int, unet_cfg.dim_mults[1:-1].split(', '))),
        flash_attn=unet_cfg.flash_attn
    )
    model = model.to(device)
    diffusion = BadDiffusion(
        model,
        image_size=diff_cfg.image_size,
        timesteps=diff_cfg.timesteps,  # number of steps
        sampling_timesteps=diff_cfg.sampling_timesteps,
        objective=diff_cfg.objective,
        trigger=triger,
        factor_list=ast.literal_eval(diff_cfg.factor_list),
        device=device,
    )

    trainer = BadTrainer(
        diffusion,
        bad_folder=trainer_cfg.bad_folder,
        good_folder=trainer_cfg.good_folder,
        train_batch_size=trainer_cfg.train_batch_size,
        train_lr=trainer_cfg.train_lr,
        train_num_steps=trainer_cfg.train_num_steps,
        gradient_accumulate_every=trainer_cfg.gradient_accumulate_every,
        ema_decay=trainer_cfg.ema_decay,
        amp=trainer_cfg.amp,
        calculate_fid=trainer_cfg.calculate_fid,
        ratio=trainer_cfg.ratio,
        results_folder=trainer_cfg.results_folder,
        server=trainer_cfg.server,
        save_and_sample_every=trainer_cfg.save_and_sample_every if trainer_cfg.save_and_sample_every > 0 else trainer_cfg.train_num_steps,
    )

    loss_list = trainer.train()
    ret = {
        'loss_list': loss_list,
        'config': cfg,
        'diffusion': diffusion.state_dict(),
    }
    torch.save(ret, str(trainer_cfg.results_folder / f'ret.pth'))
    tg_bot.send2bot(cfg, trainer_cfg.server)


if __name__ == '__main__':
    main()
