import os
from datetime import timedelta
from functools import partial

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
from tools.time import now
from tools.prepare_data import prepare_bad_data
from tools.dataset import rm_if_exist, save_tensor_images
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

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
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
    prepare_bad_data(config)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))
    import os
    import shutil
    script_name = os.path.basename(__file__)
    target_folder = f'../results/{config.attack}/{config.dataset_name}/{now()}'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    target_file_path = os.path.join(target_folder, script_name)
    shutil.copy(__file__, target_file_path)
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
        dropout=config.unet.dropout
    ).to(device)
    trans = Compose([
        ToTensor(), Resize((config.image_size, config.image_size))
    ])
    all_path = f'../dataset/dataset-{config.dataset_name}-all'
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = InceptionV3([block_idx])
    fid_model.to(config.device)
    m1, s1 = compute_statistics_of_path(all_path, fid_model, fid_estimate_batch_size, 2048, config.device, 8)
    all_data = SanDataset(
        root_dir=all_path,
        transform=trans
    )
    all_loader = DataLoader(
        dataset=all_data, batch_size=config.batch, shuffle=False, pin_memory=True, num_workers=8,
    )
    all_loader = cycle(all_loader)
    optimizer = Adam(unet.parameters(), lr)
    if loss_type == 'l1':
        loss_fn = F.l1_loss
    elif loss_type == 'l2':
        loss_fn = F.mse_loss
    else:
        raise NotImplementedError
    current_epoch = 0
    fid_value = 0
    loss_list = []
    fid_list = []
    tag = f'{config.dataset_name}_{config.attack}'
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
    with tqdm(initial=current_epoch, total=epoch) as pbar:
        while current_epoch < epoch:
            x_0 = next(all_loader)
            b, c, w, h = x_0.shape
            x_0 = x_0.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, 1000, (b,), device=device, dtype=torch.long)
            eps = torch.randn_like(x_0, device=device)
            x_t = diffusion.q_sample(x_0, t, eps)
            eps_theta = diffusion.eps_model(x_t, t)
            loss = loss_fn(eps_theta, eps)
            loss.backward()
            loss_list.append(float(loss))
            writer1.add_scalar(tag, float(loss), epoch)
            optimizer.step()
            pbar.set_description(f'loss: {loss:.4f}, fid: {fid_value:4f}')
            if current_epoch >= save_epoch and current_epoch % save_epoch == 0:
                diffusion.eps_model.eval()
                with torch.no_grad():
                    fake_sample = sample_fn(1000)
                    rm_if_exist(f'{target_folder}/fid')
                    save_tensor_images(fake_sample, f'{target_folder}/fid')
                    m2, s2 = compute_statistics_of_path(f'{target_folder}/fid', fid_model, fid_estimate_batch_size,
                                                        2048, config.device, 8)
                    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
                    fid_list.append(fid_value)
                    writer2.add_scalar(tag, float(fid_value), epoch)
                    writer2.flush()
            writer1.flush()
            del loss, x_0, x_t, eps_theta, eps
            torch.cuda.empty_cache()
            current_epoch += 1
            pbar.update(1)
    rm_if_exist(f'{target_folder}/fid')
    for i in range(int(num_fid_sample / fid_estimate_batch_size)):
        fake_sample = sample_fn(fid_estimate_batch_size)
        save_tensor_images(fake_sample, f'{target_folder}/fid')
    m2, s2 = compute_statistics_of_path(f'{target_folder}/fid', fid_model, fid_estimate_batch_size, 2048, device, 8)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    fid_list.append(fid_value)
    res = {
        'unet': unet.state_dict(),
        'opt': optimizer.state_dict(),
        "config": OmegaConf.to_object(config),
        "loss_list": loss_list,
        'fid_list': fid_list
    }
    torch.save(res, f'{target_folder}/result.pth')
    send2bot(OmegaConf.to_yaml(OmegaConf.to_object(config)), 'over')
    print(target_folder)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train()
