import time
import os
import shutil
from random import random
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
from tools.unet import Unet
from tools.time import now, get_hour
from tools.prepare_data import prepare_bad_data
from tools.dataset import rm_if_exist, save_tensor_images, load_dataloader
from tools.tg_bot import send2bot


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class SanDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, sample_step):
        super().__init__()
        self.eps_model = eps_model
        self.sample_step = sample_step
        self.device = device
        self.image_size = self.eps_model.image_size
        self.n_steps = n_steps
        self.ema = EMA(self.eps_model, update_every=10)
        self.ema.to(device=self.device)

        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        # self.beta = sigmoid_beta_schedule(self.n_steps).to(device)
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
        eps_theta = self.ema.ema_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps
        # return mean + (0.5 * gather(self.posterior_log_variance_clipped, t)).exp() * eps

    def pred_x_0_form_eps_theta(self, x_t, eps_theta, t):
        return (gather(torch.sqrt(1. / self.alpha_bar), t) * x_t -
                gather(torch.sqrt(1. / self.alpha_bar - 1), t) * eps_theta)

    @torch.inference_mode()
    def ddpm_sample(self, batch):
        x_t = torch.randn(size=(batch, self.eps_model.channel, self.image_size, self.image_size), device=self.device)
        for i in tqdm(reversed(range(0, self.n_steps)), desc='DDPM Sampling', total=self.n_steps, leave=False):
            x_t_m_1 = self.p_sample(x_t, torch.tensor([i], device=self.device))
            x_t = x_t_m_1
        return x_t

    @torch.inference_mode()
    def ddim_sample(self, batch):
        batch, device, total_timesteps, sampling_timesteps, eta = batch, self.device, self.n_steps, 250, 0
        shape = (batch, self.eps_model.channel, self.image_size, self.image_size)
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)
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
    if loss_type == 'l1':
        loss_fn = F.l1_loss
    elif loss_type == 'l2':
        loss_fn = F.mse_loss
    else:
        raise NotImplementedError
    current_epoch = 0
    diffusion = SanDiffusion(unet, config.diffusion.timesteps, device, sample_step=config.diffusion.sampling_timesteps)
    if torch.cuda.device_count() > 1 and config.batch > 256:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device_ids = [0, 1]
        diffusion.eps_model = nn.DataParallel(diffusion.eps_model, device_ids=device_ids).to('cuda')
    if sample_type == 'ddim':
        sample_fn = diffusion.ddim_sample
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
        if config.attack == "badnet":
            trigger_path = f'../resource/badnet/trigger_{config.image_size}_{int(config.image_size / 10)}.png'
            mask_path = f'../resource/badnet/mask_{config.image_size}_{int(config.image_size / 10)}.png'
            mask = trans(Image.open(mask_path))
            mask = mask.to(device)
        elif config.attack == 'blended':
            trigger_path = '../resource/blended/hello_kitty.jpeg'
        else:
            raise NotImplementedError
        trigger = trans(Image.open(trigger_path))
        trigger = trigger.to(device)
        gamma = config.gamma
        assert config.p_start < config.p_end
    loss_list = []
    mode = 'b'
    with tqdm(initial=current_epoch, total=epoch) as pbar:
        while current_epoch < epoch:
            if config.attack != 'benign':
                if random() < config.ratio:
                    x_0 = next(bad_loader)
                    b, c, w, h = x_0.shape
                    t = torch.randint(config.p_start, config.p_end, (b,), device=device, dtype=torch.long)
                    mode = 'p'
                else:
                    x_0 = next(good_loader)
                    b, c, w, h = x_0.shape
                    t = torch.randint(0, 1000, (b,), device=device, dtype=torch.long)
                    mode = 'b'
            else:
                x_0 = next(all_loader)
                b, c, w, h = x_0.shape
                t = torch.randint(0, 1000, (b,), device=device, dtype=torch.long)
            x_0 = x_0.to(device)
            x_0 = normalize_to_neg_one_to_one(x_0)
            optimizer.zero_grad()
            eps = torch.randn_like(x_0, device=device)
            x_t = diffusion.q_sample(x_0, t, eps)
            eps_theta = diffusion.eps_model(x_t, t)
            loss = loss_fn(eps_theta, eps)
            if config.attack != 'benign':
                loss += loss_fn(eps_theta, eps - trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1) * gamma)
            loss.backward()
            optimizer.step()
            diffusion.ema.update()
            pbar.set_description(f'{mode} loss: {loss:.5f}')
            loss_list.append(float(loss))
            if current_epoch >= save_epoch and current_epoch % save_epoch == 0:
                diffusion.ema.ema_model.eval()
                with torch.inference_mode():
                    rm_if_exist(f'{target_folder}/fid')
                    fake_sample = sample_fn(16)
                    torchvision.utils.save_image(fake_sample, f'{target_folder}/sample_{current_epoch}.png', nrow=4)
            current_hour = get_hour()
            if current_hour in range(10, 21) and config.server == "lab":
                time.sleep(0.1)
            current_epoch += 1
            pbar.update(1)
    res = {
        'unet': unet.state_dict(),
        'opt': optimizer.state_dict(),
        'ema': diffusion.ema.state_dict(),
        "config": OmegaConf.to_object(config),
        'loss_list': loss_list,
    }
    torch.save(res, f'{target_folder}/result.pth')
    send2bot(OmegaConf.to_yaml(OmegaConf.to_object(config)), 'over')
    print(target_folder)


if __name__ == '__main__':
    train()
