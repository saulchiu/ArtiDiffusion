import argparse
import ast
import math
import pdb
import time

import PIL.Image
import denoising_diffusion_pytorch
import torch
import torchvision.transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch.nn.functional as F
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, rearrange, random, reduce, extract, cycle, \
    Dataset, divisible_by, num_to_groups, exists
from PIL import Image
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('../')
from tools import tg_bot
from tools.img import cal_ssim
from tools.prepare_data import prepare_bad_data
from tools.time import now


class BadDiffusion(GaussianDiffusion):
    @property
    def device(self):
        return self._device

    def __init__(self, model, image_size, timesteps, sampling_timesteps, objective, trigger, device, attack, gamma):
        super().__init__(model, image_size=image_size, timesteps=timesteps, sampling_timesteps=sampling_timesteps,
                         objective=objective)
        self.trigger = trigger
        self.attack = attack
        self.device = device
        self.gamma = gamma

    def train_mode_p_sample(self, x, t, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def bad_p_loss(self, x_start, t, mode, noise=None, offset_noise_strength=None):
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
            # optimize the epsilon_{no trigger}
            loss_1 = F.mse_loss(target, model_out, reduction='none')
            loss_1 = reduce(loss_1, 'b ... -> b', 'mean')
            loss_1 = loss_1 * extract(self.loss_weight, t, loss_1.shape)
            loss_1 = loss_1.mean()
            x_t = x
            loss_2 = 0
            if self.attack == "badnet":
                loss_2 = self.badnet_loss(x_start, x_t, model_out, t, target)
            elif self.attack == "blended":
                loss_2 = self.blended_loss(x_start, x_t, model_out, t, target)
            loss = loss_1 + loss_2
            # print(loss)
            if math.isnan(float(loss)):
                print("Loss is NaN!")
                # pdb.set_trace()
        return loss

    def badnet_loss(self, x_start, x_t, epsilon_p, t, target):
        tg = self.trigger.unsqueeze(0).expand(x_t.shape[0], -1, -1, -1)
        tg = tg.to(x_start.device)
        loss_2 = F.mse_loss(epsilon_p, target - tg * self.gamma)
        return loss_2

    def blended_loss(self, x_start, x_t, epsilon_p, t, target):
        tg = self.trigger.unsqueeze(0).expand(x_t.shape[0], -1, -1, -1)
        tg = tg.to(epsilon_p.device)
        z = torch.randn_like(x_start)
        # factor = extract(self.betas, t, x_start.shape) / extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        loss_2 = F.mse_loss(epsilon_p, target - tg * self.gamma)
        return loss_2

    def forward(self, img, mode, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        if mode == 0:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        else:
            t = torch.randint(200, 400, (b,), device=device).long()
        img = self.normalize(img)
        return self.bad_p_loss(img, t, mode, *args, **kwargs)

    @device.setter
    def device(self, value):
        self._device = value


class BadTrainer(denoising_diffusion_pytorch.Trainer):
    def __init__(self, diffusion, good_folder, train_batch_size, train_lr, train_num_steps,
                 gradient_accumulate_every, ratio, results_folder, save_and_sample_every, ema_decay, amp,
                 calculate_fid, bad_folder, all_folder):
        super().__init__(diffusion_model=diffusion, folder=all_folder, train_batch_size=train_batch_size,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         gradient_accumulate_every=gradient_accumulate_every, ema_decay=ema_decay, amp=amp,
                         calculate_fid=calculate_fid, results_folder=results_folder,
                         save_and_sample_every=save_and_sample_every)
        self.ratio = ratio
        if bad_folder is not None:
            self.bad_ds = Dataset(bad_folder, self.image_size, augment_horizontal_flip=True, convert_image_to='RGB')
            bad_dl = DataLoader(self.bad_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                num_workers=4)
            bad_dl = self.accelerator.prepare(bad_dl)
            self.bad_dl = cycle(bad_dl)
        if good_folder is not None:
            self.good_ds = Dataset(good_folder, self.image_size, augment_horizontal_flip=True, convert_image_to='RGB')
            good_dl = DataLoader(self.good_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)
            good_dl = self.accelerator.prepare(good_dl)
            self.good_dl = cycle(good_dl)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'diffusion': self.accelerator.get_state_dict(self.model),
            'unet': self.model.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        loss_list = []
        fid_list = []
        min_loss = 1e3
        min_fid = 1e3
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for i in range(self.gradient_accumulate_every):
                    import random
                    if random.random() < self.ratio:
                        data = next(self.bad_dl).to(device)
                        mode = 1  # poisoning
                    else:
                        data = next(self.good_dl).to(device)
                        mode = 0
                    with self.accelerator.autocast():
                        loss = self.model(data, mode)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                pbar.set_description(f'loss: {total_loss:.7f}')
                formatted_loss = format(total_loss, '.7f')
                min_loss = min(min_loss, total_loss)
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

                        torchvision.utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                                     nrow=int(math.sqrt(self.num_samples)))
                        self.save(milestone)
                        # whether to calculate fid
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            fid_list.append(fid_score)
                            min_fid = min(fid_score, min_fid)
                            tg_bot.send2bot(msg=f'min loss: {min_loss};\n min fid: {min_fid}', title='status')
                            if min_loss < 1e-3 or min_fid < 10:
                                print(self.step)
                                break
                            accelerator.print(f'fid_score: {fid_score}')
                pbar.update(1)
        accelerator.print('training complete')
        return loss_list, fid_list


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))
    unet_cfg = config.noise_predictor
    diff_cfg = config.diffusion
    trainer_cfg = config.trainer
    import os
    import shutil
    target_folder = f'../results/{config.attack}/{config.dataset_name}/{now()}'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    script_name = os.path.basename(__file__)
    target_file_path = os.path.join(target_folder, script_name)
    shutil.copy(__file__, target_file_path)
    device = diff_cfg.device
    import os
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    trigger_path = f'../resource/{config.attack}/{config.trigger_name}'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    trigger = Image.open(trigger_path)
    trigger = transform(trigger)
    trigger = trigger.to(device)
    prepare_bad_data(config)
    unet = Unet(
        dim=unet_cfg.dim,
        dim_mults=tuple(map(int, unet_cfg.dim_mults[1:-1].split(', '))),
        flash_attn=unet_cfg.flash_attn
    )
    diffusion = BadDiffusion(
        unet,
        image_size=diff_cfg.image_size,
        timesteps=diff_cfg.timesteps,  # number of steps
        sampling_timesteps=diff_cfg.sampling_timesteps,
        objective=diff_cfg.objective,
        trigger=trigger,
        device=device,
        attack=diff_cfg.attack,
        gamma=diff_cfg.gamma
    )
    ratio = config.trainer.ratio
    dataset_all = f'../dataset/dataset-{config.dataset_name}-all'
    dataset_bad = f'../dataset/dataset-{config.dataset_name}-bad-{config.attack}-{str(ratio)}'
    dataset_good = f'../dataset/dataset-{config.dataset_name}-good-{config.attack}-{str(ratio)}'
    trainer = BadTrainer(
        diffusion,
        bad_folder=dataset_bad,
        good_folder=dataset_good,
        all_folder=dataset_all,
        train_batch_size=trainer_cfg.train_batch_size,
        train_lr=trainer_cfg.train_lr,
        train_num_steps=trainer_cfg.train_num_steps,
        gradient_accumulate_every=trainer_cfg.gradient_accumulate_every,
        ema_decay=trainer_cfg.ema_decay,
        amp=trainer_cfg.amp,
        calculate_fid=trainer_cfg.calculate_fid,
        ratio=trainer_cfg.ratio,
        results_folder=target_folder,
        save_and_sample_every=trainer_cfg.save_and_sample_every if trainer_cfg.save_and_sample_every > 0 else trainer_cfg.train_num_steps,
    )
    loss_list, fid_list = trainer.train()
    if trainer.accelerator.is_main_process:
        ret = {
            'loss_list': loss_list,
            'fid_list': fid_list,
            'config': OmegaConf.to_object(config),
            'unet': unet.state_dict(),
            'diffusion': diffusion.state_dict(),
        }
        torch.save(ret, f'{target_folder}/result.pth')
        tg_bot.send2bot(OmegaConf.to_yaml(OmegaConf.to_object(config)), 'over')
        print(target_folder)


if __name__ == '__main__':
    main()
