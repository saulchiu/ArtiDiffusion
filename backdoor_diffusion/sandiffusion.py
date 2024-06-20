import os
import time
from datetime import timedelta
from functools import partial
import os
import shutil
from random import random

import torchvision.utils
import yaml
from accelerate import accelerator
from pytorch_fid.fid_score import calculate_frechet_distance, compute_statistics_of_path
from pytorch_fid.inception import InceptionV3

import torch
from PIL import Image
from labml_nn.diffusion.ddpm import DenoiseDiffusion, experiment
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose, ToTensor, Resize
from torch.optim.adam import Adam
import torch.nn.functional as F
from torchvision.utils import save_image
from ema_pytorch.ema_pytorch import EMA
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import sys

sys.path.append('../')
from tools.unet import Unet
from tools.dataset import cycle, SanDataset
from tools.samper import DDIM_Sampler
from tools.time import now, get_hour
from tools.prepare_data import prepare_bad_data
from tools.dataset import rm_if_exist, save_tensor_images, load_dataloader
from tools.tg_bot import send2bot


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class SanDiffusion(DenoiseDiffusion):
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, sample_step):
        super().__init__(eps_model, n_steps, device)
        self.sample_step = sample_step
        self.device = self.eps_model.device
        self.image_size = self.eps_model.image_size
        self.ema = EMA(self.eps_model, update_every=10)
        self.ema.to(device=self.device)

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.ema.ema_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps

    @torch.inference_mode()
    def ddpm_sample(self, batch):
        x_t = torch.randn(size=(batch, self.eps_model.channel, self.image_size, self.image_size), device=self.device)
        for i in tqdm(reversed(range(0, self.n_steps)), desc='DDPM Sampling', total=self.n_steps, leave=False):
            x_t_m_1 = self.p_sample(x_t, torch.tensor([i], device=self.device))
            x_t = x_t_m_1
        return x_t


@hydra.main(version_base=None, config_path='../config', config_name='default')
def train(config: DictConfig):
    """
    prepare dataset, save source code, save config file
    """
    prepare_bad_data(config)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))
    script_name = os.path.basename(__file__)
    target_folder = f'../results/{config.attack}/{config.dataset_name}/{now()}'
    print(target_folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    target_file_path = os.path.join(target_folder, script_name)
    shutil.copy(__file__, target_file_path)
    with open(f'{target_folder}/config.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_object(config), f, allow_unicode=True)
    """
    load config
    """
    device = config.device
    lr = config.lr
    loss_type = config.loss_type
    sample_type = config.sample_type
    save_epoch = config.save_epoch
    epoch = config.epoch
    num_fid_sample = 5e4
    fid_estimate_batch_size = config.fid_estimate_batch_size
    unet = Unet(
        dim=config.unet.dim,
        image_size=config.image_size,
        dim_multiply=tuple(map(int, config.unet.dim_mults[1:-1].split(', '))),
        dropout=config.unet.dropout,
        device=device
    )
    trans = Compose([
        ToTensor(), Resize((config.image_size, config.image_size))
    ])
    all_path = f'../dataset/dataset-{config.dataset_name}-all'
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    with torch.no_grad():
        fid_model = InceptionV3([block_idx])
        fid_model.to(config.device)
        m1, s1 = compute_statistics_of_path(all_path, fid_model, fid_estimate_batch_size, 2048, config.device, 8)
    all_loader = load_dataloader(path=all_path, trans=trans, batch=config.batch)
    optimizer = Adam(unet.parameters(), lr)
    if loss_type == 'l1':
        loss_fn = F.l1_loss
    elif loss_type == 'l2':
        loss_fn = F.mse_loss
    else:
        raise NotImplementedError
    current_epoch = 0
    fid_value = 0
    tag = f'{config.dataset_name}_{config.attack}_{str(config.ratio)}'
    rm_if_exist(f'../runs/{tag}_loss')
    rm_if_exist(f'../runs/{tag}_fid')
    writer1 = SummaryWriter(f'../runs/{tag}_loss')
    writer2 = SummaryWriter(f'../runs/{tag}_fid')
    diffusion = SanDiffusion(unet, config.diffusion.timesteps, device, sample_step=config.diffusion.sampling_timesteps)
    if sample_type == 'ddim':
        samper = DDIM_Sampler(diffusion)
        sample_fn = samper.sample
    elif sample_type == 'ddpm':
        sample_fn = diffusion.ddpm_sample
    else:
        raise NotImplementedError
    """
    prepare poisoning
    """
    if config.attack != "benign":
        ratio = config.ratio
        bad_path = f'../dataset/dataset-{config.dataset_name}-bad-{config.attack}-{str(ratio)}'
        good_path = f'../dataset/dataset-{config.dataset_name}-good-{config.attack}-{str(ratio)}'
        bad_loader = load_dataloader(bad_path, trans, config.batch)
        good_loader = load_dataloader(good_path, trans, config.batch)
        if config.attack =="badnet":
            trigger_path = f'../resource/badnet/trigger_{config.image_size}_{int(config.image_size / 10)}.png'
        elif config.attack == 'blended':
            trigger_path = '../resource/blended/hello_kitty.jpeg'
        else:
            raise NotImplementedError
        trigger = trans(Image.open(trigger_path))
        trigger = trigger.to(device)
        gamma = config.gamma
    with tqdm(initial=current_epoch, total=epoch) as pbar:
        while current_epoch < epoch:
            if config.attack != 'benign':
                if random() < config.ratio:
                    x_0 = next(bad_loader)
                    b, c, w, h = x_0.shape
                    t = torch.randint(200, 400, (b,), device=device, dtype=torch.long)
                else:
                    x_0 = next(good_loader)
                    b, c, w, h = x_0.shape
                    t = torch.randint(0, 1000, (b,), device=device, dtype=torch.long)
            else:
                x_0 = next(all_loader)
                b, c, w, h = x_0.shape
                t = torch.randint(0, 1000, (b,), device=device, dtype=torch.long)
            x_0 = x_0.to(device)
            optimizer.zero_grad()
            eps = torch.randn_like(x_0, device=device)
            x_t = diffusion.q_sample(x_0, t, eps)
            # no need to use ema
            eps_theta = diffusion.eps_model(x_t, t)
            loss = loss_fn(eps_theta, eps)
            if config.attack != 'benign':
                loss /= 2
                loss += loss_fn(eps_theta, eps - trigger.unsqueeze(0).expand(eps.shape[0], -1, -1, -1) * gamma) / 2
            loss.backward()
            writer1.add_scalar(tag, float(loss), current_epoch)
            optimizer.step()
            diffusion.ema.update()
            pbar.set_description(f'loss: {loss:.4f}, fid: {fid_value:4f}')
            if current_epoch >= save_epoch and current_epoch % save_epoch == 0:
                diffusion.ema.ema_model.eval()
                with torch.inference_mode():
                    rm_if_exist(f'{target_folder}/fid')
                    for i in range(int(fid_estimate_batch_size / 64)):
                        fake_sample = sample_fn(64)
                        save_tensor_images(fake_sample, f'{target_folder}/fid')
                    torchvision.utils.save_image(fake_sample, f'{target_folder}/sample_{current_epoch}.png', nrow=8)
                    m2, s2 = compute_statistics_of_path(f'{target_folder}/fid', fid_model, fid_estimate_batch_size,
                                                        2048, config.device, 8)
                    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
                    writer2.add_scalar(tag, float(fid_value), current_epoch)
                    writer2.flush()
            writer1.flush()
            current_hour = get_hour()
            # if (current_hour in range(0, 10) or current_hour in range(21, 24)) == False and config.server == "lab":
            if current_hour in range(10, 21) and config.server == "lab":
                time.sleep(0.1)
                # del loss, x_0, x_t, t, eps, eps_theta
                # torch.cuda.empty_cache()
            current_epoch += 1
            pbar.update(1)
    rm_if_exist(f'{target_folder}/fid')
    with torch.inference_mode():
        loop = int(num_fid_sample / fid_estimate_batch_size)
        for i in tqdm(range(loop), desc='Generating FID samples'):
            fake_sample = sample_fn(fid_estimate_batch_size)
            save_tensor_images(fake_sample, f'{target_folder}/fid')
        if num_fid_sample - loop * fid_estimate_batch_size != 0:
            fake_sample = sample_fn(num_fid_sample - loop * fid_estimate_batch_size)
            save_tensor_images(fake_sample, f'{target_folder}/fid')
        m2, s2 = compute_statistics_of_path(f'{target_folder}/fid', fid_model, fid_estimate_batch_size, 2048, device, 8)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    res = {
        'unet': unet.state_dict(),
        'opt': optimizer.state_dict(),
        'ema': diffusion.ema.state_dict(),
        "config": OmegaConf.to_object(config),
        'fid: ': float(fid_value)
    }
    torch.save(res, f'{target_folder}/result.pth')
    send2bot(OmegaConf.to_yaml(OmegaConf.to_object(config)), 'over')
    print(target_folder)


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    train()
