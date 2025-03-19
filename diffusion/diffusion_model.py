import math
import time
import os
import shutil
from random import random

import numpy as np
import torchvision.utils
import yaml
import torch
from PIL import Image
from typing import Tuple, Optional
from torch import nn
from torchvision.transforms.transforms import Compose, ToTensor, Resize
from torch.optim.adam import Adam
import torch.nn.functional as F
from ema_pytorch.ema_pytorch import EMA
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import sys

sys.path.append('../')
from diffusion.unet import Unet
from tools.time import now, get_hour
from tools.prepare_data import prepare_bad_data
from tools.dataset import rm_if_exist, load_dataloader
from tools.ftrojann_transform import get_ftrojan_transform
from tools.ctrl_transform import ctrl
from tools.utils import unsqueeze_expand
from tools.inject_backdoor import get_artifact


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)





def get_beta_schedule(beta_schedule, beta_start, beta_end, n_steps):
    if beta_schedule == 'linear':
        beta = torch.linspace(beta_start, beta_end, n_steps)
    elif beta_schedule == 'sigmoid':
        beta = torch.linspace(-6, 6, n_steps)
        beta = torch.sigmoid(beta) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'cosine':
        def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            return np.array(betas)

        beta = betas_for_alpha_bar(
            n_steps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
        beta = torch.from_numpy(beta)
    elif beta_schedule == 'scaled_linear':
        beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_steps) ** 2
    elif beta_schedule == "squaredcos_cap_v2":
        def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
            if alpha_transform_type == "cosine":
                def alpha_bar_fn(t):
                    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            elif alpha_transform_type == "exp":
                def alpha_bar_fn(t):
                    return math.exp(t * -12.0)
            else:
                raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
            return torch.tensor(betas, dtype=torch.float32)
        # Glide cosine schedule
        beta = betas_for_alpha_bar(n_steps)
    elif beta_schedule == "const":
        beta = beta_end * np.ones(n_steps, dtype=np.float64)
        beta = torch.from_numpy(beta)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        beta = 1.0 / np.linspace(
            n_steps, 1, n_steps, dtype=np.float64
        )
        beta = torch.from_numpy(beta)
    else:
        raise NotImplementedError(beta_schedule)
    return beta.float()


class DiffusionModel:
    def __init__(self, eps_model: Unet, n_steps: int, device, sample_step, beta_schedule, beta_start, beta_end):
        super().__init__()
        self.eps_model = eps_model
        self.sample_step = sample_step
        self.device = device
        self.image_size = self.eps_model.image_size
        self.n_steps = n_steps
        self.ema = EMA(self.eps_model, update_every=10)
        self.ema.to(device=self.device)
        self.beta_schedule = beta_schedule
        # beta = get_beta_schedule(beta_schedule, 1e-4, 2e-2, n_steps)
        beta = get_beta_schedule(beta_schedule, beta_start, beta_end, n_steps)
        self.beta = beta.to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta
        self.alphas_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.)
        self.posterior_variance = self.beta * (1. - self.alphas_bar_prev) / (1. - self.alpha_bar)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    @torch.inference_mode()
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        # eps_theta = self.ema.ema_model(xt, t)
        # alpha_bar = gather(self.alpha_bar, t)
        # alpha = gather(self.alpha, t)
        # eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # var = gather(self.sigma2, t)
        # eps = torch.randn(xt.shape, device=xt.device)
        # return mean + (var ** .5) * eps
        eps_theta = self.ema.ema_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / ((1 - alpha_bar) ** .5)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        z = torch.randn_like(xt) if t > 0 else 0.
        return mean + (0.5 * gather(self.posterior_log_variance_clipped, t)).exp() * z

    def pred_x_0_form_eps_theta(self, x_t, eps_theta, t, clip=False):
        tiled_x_0 = (gather(torch.sqrt(1. / self.alpha_bar), t) * x_t -
                     gather(torch.sqrt(1. / self.alpha_bar - 1), t) * eps_theta)
        if clip:
            tiled_x_0 = torch.clip(tiled_x_0, min=-1, max=1)
        return tiled_x_0

    @torch.inference_mode()
    def ddpm_sample(self, batch, x_t=None, sampling_timesteps=None):
        x_t = torch.randn(size=(batch, self.eps_model.channel, self.image_size, self.image_size),
                          device=self.device) if x_t is None else x_t
        sampling_timesteps = self.n_steps if sampling_timesteps == None else sampling_timesteps
        for i in tqdm(reversed(range(1, sampling_timesteps)), desc='DDPM Sampling', total=sampling_timesteps, leave=False):
            x_t_m_1 = self.p_sample(x_t, torch.tensor([i], device=self.device))
            x_t = x_t_m_1
        return x_t

    @torch.inference_mode()
    def ddim_sample(self, batch, img=None, sampling_timesteps=250):
        batch, device, total_timesteps, sampling_timesteps, eta = batch, self.device, self.n_steps, sampling_timesteps, 0
        shape = (batch, self.eps_model.channel, self.image_size, self.image_size)
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device) if img is None else img
        imgs = [img]
        for time, time_next in tqdm(time_pairs, desc='DDIM sample'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise = self.ema.ema_model(img, time_cond)
            x_start = (gather(torch.sqrt(1. / self.alpha_bar), time_cond) * img -
                       gather(torch.sqrt(1. / self.alpha_bar - 1), time_cond) * pred_noise)
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue
            alpha = self.alpha_bar[time]
            alpha_next = self.alpha_bar[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            imgs.append(img)
        return img

    @torch.inference_mode()
    def dpm_solver_sample(self, batch):
        def update_with_dpm_solver(img, pred_noise, learning_rate):
            update = learning_rate * (img - pred_noise)
            img = img + update
            return img

        shape = (batch, self.eps_model.channel, self.image_size, self.image_size)
        img = torch.randn(shape, device=self.device)
        num_steps = 20
        learning_rate = 0.01
        for step in range(num_steps):
            pred_noise = self.ema.ema_model(img, torch.tensor([self.n_steps - step - 1] * batch, device=self.device))
            img = update_with_dpm_solver(img, pred_noise, learning_rate)
        return img

    @torch.inference_mode()
    def sample(self, batch):
        return None


@hydra.main(version_base=None, config_path='../config', config_name='default')
def train(config: DictConfig):
    """
    prepare dataset, save source code
    """
    prepare_bad_data(config)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))
    prepare_data_file = '../tools/prepare_data.py'
    target_folder = f'../results/{config.attack.name}/{config.dataset_name}/{now()}' if config.path == 'None' else config.path
    print(target_folder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    main_target_path = os.path.join(target_folder, 'sandiffusion.py')
    data_target_path = os.path.join(target_folder, 'prepare_data.py')
    shutil.copy(__file__, main_target_path)
    shutil.copy(prepare_data_file, data_target_path)
    """
    load config
    """
    device = config.device
    lr = config.lr
    save_epoch = config.save_epoch
    sample_epoch = config.sample_epoch
    epoch = config.epoch
    unet = Unet(
        dim=config.unet.dim,
        image_size=config.image_size,
        dim_multiply=tuple(map(int, config.unet.dim_mults[1:-1].split(', '))),
        dropout=config.unet.dropout,
        device=device
    )
    unet.to(device)
    trans = Compose([
        ToTensor(), Resize((config.image_size, config.image_size))
    ])
    all_path = f'../dataset/dataset-{config.dataset_name}-all'
    all_loader = load_dataloader(path=all_path, trans=trans, batch=config.batch)
    optimizer = Adam(unet.parameters(), lr)
    diffusion = DiffusionModel(
        eps_model=unet,
        n_steps=config.diffusion.timesteps,
        device=device,
        sample_step=config.diffusion.sampling_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )
    """
    prepare poisoning
    """
    if config.attack.name != "benign":
        ratio = config.ratio
        eta = config.gamma
        bad_path = f'../dataset/dataset-{config.dataset_name}-bad-{config.attack.name}-{str(ratio)}'
        good_path = f'../dataset/dataset-{config.dataset_name}-good-{config.attack.name}-{str(ratio)}'
        bad_loader = load_dataloader(bad_path, trans, config.batch)
        good_loader = load_dataloader(good_path, trans, config.batch)
        assert config.p_start < config.p_end
    def get_x_and_t():
        if config.attack.name != 'benign':
            if random() < config.ratio:
                img = next(bad_loader)
                b, c, w, h = img.shape
                time_step = torch.randint(config.p_start, config.p_end, (b,), device=device, dtype=torch.long)
                mode = 1
            else:
                img = next(good_loader)
                b, c, w, h = img.shape
                time_step = torch.randint(0, config.diffusion.timesteps, (b,), device=device, dtype=torch.long)
                mode = 0
        else:
            img = next(all_loader)
            b, c, w, h = img.shape
            time_step = torch.randint(0, config.diffusion.timesteps, (b,), device=device, dtype=torch.long)
            mode = 0
        return img.to(device), time_step, mode

    grad_acc = config.grad_acc
    current_epoch = 0
    if config.path != 'None':
        # from pth
        ld = torch.load(f'{config.path}/result.pth', map_location=device)
        # print(ld)
        current_epoch = ld['current_epoch']
        optimizer.load_state_dict(ld['opt'])
        diffusion.eps_model.load_state_dict(ld['unet'])
        diffusion.ema.load_state_dict(ld['ema'])
        del ld
    # use data parallel or not
    if torch.cuda.device_count() > 1 and config.parallel == 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device_ids = [0, 1]
        diffusion.eps_model = nn.DataParallel(diffusion.eps_model, device_ids=device_ids).to('cuda')
    # use ddpm or ddim
    if config.sample_type == 'ddim':
        sample_fn = diffusion.ddim_sample
    elif config.sample_type == 'ddpm':
        sample_fn = diffusion.ddpm_sample
    else:
        raise NotImplementedError(config.sample_type)
    if config.loss_type == 'l1':
        loss_fn = F.l1_loss
    elif config.loss_type == 'l2':
        loss_fn = F.mse_loss
    else:
        raise NotImplementedError(config.loss_type)
    '''
    start train!
    '''
    with tqdm(initial=current_epoch, total=epoch) as pbar:
        while current_epoch < epoch:
            total_loss = torch.zeros(size=(), device=device)
            optimizer.zero_grad()
            for _ in range(grad_acc):
                x_0, t, mode = get_x_and_t()
                eps = torch.randn_like(x_0, device=device)
                x_t = diffusion.q_sample(x_0, t, eps)
                eps_theta = diffusion.eps_model(x_t, t)
                loss = loss_fn(eps_theta, eps)
                if config.artifact.name != 'benign' and mode == 1:
                    artifact = get_artifact(config=config).to(device)
                    loss += loss_fn(eps_theta, eps - artifact.unsqueeze(0).expand(eps.shape[0], -1, -1, -1) * eta)
                total_loss += loss / grad_acc
            total_loss.backward()
            optimizer.step()
            diffusion.ema.update()
            pbar.set_description(f'loss: {total_loss:.5f}')
            if current_epoch >= save_epoch and current_epoch % save_epoch == 0:
                diffusion.ema.ema_model.eval()
                with torch.inference_mode():
                    print(f"save model to: {target_folder}")
                    res = {
                        'unet': unet.state_dict(),
                        'opt': optimizer.state_dict(),
                        'ema': diffusion.ema.state_dict(),
                        "config": OmegaConf.to_object(config),
                        "current_epoch": current_epoch,
                    }
                    with open(f'{target_folder}/config.yaml', 'w') as f:
                        yaml.dump(OmegaConf.to_object(config), f, allow_unicode=True)
                    torch.save(res, f'{target_folder}/result.pth')
                    del res
                diffusion.ema.ema_model.train()
            if current_epoch >= sample_epoch and current_epoch % sample_epoch == 0:
                diffusion.ema.ema_model.eval()
                with torch.inference_mode():
                    fake_sample = sample_fn(config.sample_num)
                    torchvision.utils.save_image(fake_sample, f'{target_folder}/sample_{current_epoch}.png', nrow=2)
                    del fake_sample
                diffusion.ema.ema_model.train()
            current_epoch += 1
            # free up memory fragments of GPU
            if x_0.shape[0] != config.batch:
                del total_loss, loss, x_0, x_t, eps, eps_theta
                torch.cuda.empty_cache()
            pbar.update(1)
    res = {
        'unet': unet.state_dict(),
        'opt': optimizer.state_dict(),
        'ema': diffusion.ema.state_dict(),
        "config": OmegaConf.to_object(config),
        "current_epoch": current_epoch,
    }
    with open(f'{target_folder}/config.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_object(config), f, allow_unicode=True)
    torch.save(res, f'{target_folder}/result.pth')
    print(target_folder)


if __name__ == '__main__':
    torch.manual_seed(42)
    train()
