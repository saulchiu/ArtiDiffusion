import argparse
import math
import time

import torch
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
                 gradient_accumulate_every, results_folder, server, ema_decay, amp, calculate_fid, ):
        super().__init__(diffusion_model=diffusion, folder=good_folder, train_batch_size=train_batch_size,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         gradient_accumulate_every=gradient_accumulate_every, ema_decay=ema_decay, amp=amp,
                         calculate_fid=calculate_fid, results_folder=results_folder)
        self.server = server

    def train(self):
        accelerator = self.accelerator
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                if self.server == 'lab':
                    sleep_cat()
                total_loss = 0.
                for mode in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    with self.accelerator.autocast():
                        loss = self.model(data)
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
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the process on (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--results_folder', type=str, default='./results', help='Folder to save results')
    parser.add_argument('--server', type=str, help='which server you use, lab, pc, or lv')
    parser.add_help = True
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    batch = args.batch
    train_num_steps = args.step
    device = args.device
    results_folder = args.results_folder
    server = args.server
    import os
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    model = model.to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )
    diffusion = diffusion.to(device)
    trainer = BenignTrainer(
        diffusion,
        '../dataset/dataset-cifar10-all',
        train_batch_size=batch,
        train_lr=8e-5,
        train_num_steps=train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True,  # whether to calculate fid during training
        results_folder=results_folder,
        server=server,
    )
    trainer.train()
    tg_bot.send2bot(args, server)
