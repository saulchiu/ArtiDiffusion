import argparse
import math
import time

import hydra
import torch
import torchvision
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, denoising_diffusion_pytorch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, rearrange, random, reduce, extract, cycle, \
    Dataset, divisible_by, num_to_groups
from torch.utils.data.dataloader import DataLoader
from torchvision import utils
from tqdm import tqdm
import sys

sys.path.append('..')
from tools import tg_bot
from tools.time import get_hour, get_minute, now, sleep_cat


class BenignTrainer(denoising_diffusion_pytorch.Trainer):
    def __init__(self, diffusion, good_folder, train_batch_size, train_lr, train_num_steps,
                 gradient_accumulate_every, results_folder, server, save_and_sample_every, ema_decay, amp,
                 calculate_fid):
        super().__init__(diffusion_model=diffusion, folder=good_folder, train_batch_size=train_batch_size,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         gradient_accumulate_every=gradient_accumulate_every, ema_decay=ema_decay, amp=amp,
                         calculate_fid=calculate_fid, results_folder=results_folder,
                         save_and_sample_every=save_and_sample_every)
        self.server = server

    def train(self):
        accelerator = self.accelerator
        loss_list = []
        fid_list = []
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for mode in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(self.device)
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                pbar.set_description(f'loss: {total_loss:.4f}')
                formatted_loss = format(total_loss, '.4f')
                loss_list.append(float(formatted_loss))
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
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            fid_list.append(fid_score)
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
        return loss_list, fid_list


def get_args():
    parser = argparse.ArgumentParser(description='This script does amazing things.')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for processing')
    parser.add_argument('--step', type=int, default=10000, help='Number of steps for the diffusion model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the process on (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--results_folder', type=str, default='./results', help='Folder to save results')
    parser.add_argument('--server', type=str, help='which server you use, lab, pc, or lv')
    parser.add_argument('--save_and_sample_every', type=int, default=10000, help='save every step')
    parser.add_help = True
    return parser.parse_args()


from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg: DictConfig):
    print(cfg)
    unet_cfg = cfg.noise_predictor
    diff_cfg = cfg.diffusion
    trainer_cfg = cfg.trainer

    import os
    import shutil
    script_name = os.path.basename(__file__)
    target_folder = str(trainer_cfg.results_folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    target_file_path = os.path.join(target_folder, script_name)
    shutil.copy(__file__, target_file_path)

    device = diff_cfg.device
    import os
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    model = Unet(
        dim=unet_cfg.dim,
        dim_mults=tuple(map(int, unet_cfg.dim_mults[1:-1].split(', '))),
        flash_attn=unet_cfg.flash_attn
    )
    model = model.to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size=diff_cfg.image_size,
        timesteps=diff_cfg.timesteps,  # number of steps
        sampling_timesteps=diff_cfg.sampling_timesteps,
        objective=diff_cfg.objective,
    )

    trainer = BenignTrainer(
        diffusion,
        good_folder=trainer_cfg.all_folder,
        train_batch_size=trainer_cfg.train_batch_size,
        train_lr=trainer_cfg.train_lr,
        train_num_steps=trainer_cfg.train_num_steps,
        gradient_accumulate_every=trainer_cfg.gradient_accumulate_every,
        ema_decay=trainer_cfg.ema_decay,
        amp=trainer_cfg.amp,
        calculate_fid=trainer_cfg.calculate_fid,
        results_folder=trainer_cfg.results_folder,
        server=trainer_cfg.server,
        save_and_sample_every=trainer_cfg.save_and_sample_every if trainer_cfg.save_and_sample_every > 0 else trainer_cfg.train_num_steps,
    )
    loss_list, fid_list = trainer.train()
    ret = {
        'loss_list': loss_list,
        'fid_list': fid_list,
        'config': OmegaConf.to_yaml(OmegaConf.to_object(cfg)),
        'diffusion': diffusion.state_dict(),
    }
    torch.save(ret, f'{trainer_cfg.results_folder}/result.pth')
    tg_bot.send2bot(cfg, trainer_cfg.server)


if __name__ == '__main__':
    main()
