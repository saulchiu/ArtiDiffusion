import argparse
import ast
import math
import pdb

import PIL.Image
import denoising_diffusion_pytorch
import torch
import torchvision.transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch.nn.functional as F
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, rearrange, random, reduce, extract, cycle, \
    Dataset, divisible_by
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('../')
from tools import tg_bot
from tools.prepare_data import prepare_bad_data
from tools.time import now


class BadDiffusion(GaussianDiffusion):
    @property
    def device(self):
        return self._device

    def __init__(self, model, image_size, timesteps, sampling_timesteps, objective, trigger,
                 factor_list, device, reverse_step, attack):
        super().__init__(model, image_size=image_size, timesteps=timesteps, sampling_timesteps=sampling_timesteps,
                         objective=objective)
        self.trigger = trigger
        self.factor_list = factor_list
        self.device = device
        self.reverse_step = reverse_step
        self.attack = attack

    def train_mode_p_sample(self, x, t, x_self_cond=None):
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
            # optimize the epsilon_{no trigger}
            loss_1 = F.mse_loss(target, model_out, reduction='none')
            loss_1 = reduce(loss_1, 'b ... -> b', 'mean')
            loss_1 = loss_1 * extract(self.loss_weight, t, loss_1.shape)
            loss_1 = loss_1.mean()
            x_t = x
            loss_2 = 0
            if self.attack == "badnet":
                loss_2 = self.badnet_loss(x_start, x_t)
            elif self.attack == "blended":
                loss_2 = self.blended_loss(x_start, x_t, model_out)
            loss = self.factor_list[0] * loss_1 + self.factor_list[1] * loss_2
            # print(loss)
            if math.isnan(float(loss)):
                print("Loss is NaN!")
                # pdb.set_trace()
        return loss

    def badnet_loss(self, x_start, x_t):
        import sys
        sys.path.append('..')
        mask = PIL.Image.open('../resource/badnet/trigger_image.png')
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), torchvision.transforms.Resize((x_start.shape[2], x_start.shape[2]))
        ])
        mask = trans(mask).to(self.device)
        loss_2 = 0
        g_p = self.trigger.unsqueeze(0).expand(x_t.shape[0], -1, -1, -1)
        for i in reversed(range(self.reverse_step)):  # i is [5, 4, 3, 2, 1, 0]
            x_t_sub, _ = self.train_mode_p_sample(x_t, i + 1)
            x_t_sub.clamp_(-1., 1.)
            loss_2 += F.mse_loss(g_p, x_t_sub * mask)
            x_t = x_t_sub
        # loss_2 /= self.reverse_step
        return loss_2

    def blended_loss(self, x_start, x_t, epsilon_p):
        loss_2 = 0
        tg = self.trigger.unsqueeze(0).expand(x_t.shape[0], -1, -1, -1)
        for i in reversed(range(self.reverse_step)):
            i_t = torch.tensor(i, device=x_t.device).expand(x_t.shape[0])
            x_t_sub, _ = self.train_mode_p_sample(x_t, i + 1)
            x_t_sub.clamp_(-1., 1.)
            loss_2 += F.mse_loss(tg, (x_t_sub - 0.8 * x_start) / 0.2)
            x_t = x_t_sub
        # loss_2 /= self.reverse_step
        return loss_2

    def forward(self, img, mode, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        if mode == 0:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        else:
            t = torch.randint(self.reverse_step, 20, (1,), device=device).long()
        img = self.normalize(img)
        return self.bad_p_losses(img, t, mode, *args, **kwargs)

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
        fid_list = []
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for i in range(self.gradient_accumulate_every):
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

                    # if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    #     self.ema.ema_model.eval()
                    #     if self.calculate_fid:
                    #         fid_score = self.fid_scorer.fid_score()
                    #         accelerator.print(f'fid_score: {fid_score}')
                    #         fid_list.append(fid_score)
                pbar.update(1)
        accelerator.print('training complete')
        return loss_list, fid_list


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
    print(OmegaConf.to_yaml(OmegaConf.to_object(cfg)))
    unet_cfg = cfg.noise_predictor
    diff_cfg = cfg.diffusion
    trainer_cfg = cfg.trainer
    prepare_bad_data(cfg)
    import os
    import shutil
    script_name = os.path.basename(__file__)
    target_folder = f'../results/{cfg.attack}/{cfg.dataset_name}/{now()}'
    cfg.trainer.results_folder = target_folder
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    target_file_path = os.path.join(target_folder, script_name)
    shutil.copy(__file__, target_file_path)

    device = diff_cfg.device
    import os
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    trigger_path = diff_cfg.trigger
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    trigger = Image.open(trigger_path)
    trigger = transform(trigger).to(device)
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
        trigger=trigger,
        factor_list=ast.literal_eval(str(diff_cfg.factor_list)),
        device=device,
        reverse_step=diff_cfg.reverse_step,
        attack=diff_cfg.attack
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
        results_folder=target_folder,
        server=trainer_cfg.server,
        save_and_sample_every=trainer_cfg.save_and_sample_every if trainer_cfg.save_and_sample_every > 0 else trainer_cfg.train_num_steps,
    )
    loss_list, fid_list = trainer.train()
    ret = {
        'loss_list': loss_list,
        'fid_list': fid_list,
        'config': OmegaConf.to_object(cfg),
        'diffusion': diffusion.state_dict(),
    }
    torch.save(ret, f'{target_folder}/result.pth')
    tg_bot.send2bot(OmegaConf.to_yaml(OmegaConf.to_object(cfg)), trainer_cfg.server)


if __name__ == '__main__':
    main()
