import argparse
import math
import random

import PIL
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
import hydra
from omegaconf import OmegaConf, DictConfig
from ema_pytorch.ema_pytorch import EMA

import sys

from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append('../')
from tools.sandiffusion import SanDiffusion
from tools.dataset import save_tensor_images, rm_if_exist
from tools.prepare_data import get_dataset
from tools.unet import Unet


def gen_sample(diffusion, total_sample, target_folder):
    rm_if_exist(target_folder)
    loop = int(total_sample / 64)
    for _ in tqdm(range(loop)):
        fake_sample = diffusion.sample(64)
        save_tensor_images(fake_sample, target_folder)
    if (total_sample - loop * 64) != 0:
        fake_sample = diffusion.sample(total_sample - loop * 64)
        save_tensor_images(fake_sample, target_folder)
    return


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def plot_images(images, num_images, net=None):
    if len(images.shape) > 4:
        images = images[0]
    if net is not None:
        y_p = net(images)
        _, indexes = y_p.max(1)
    cols = int(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    figsize_width = cols * 5
    figsize_height = rows * 5
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height))
    axes = axes.flatten()
    for idx, (img, ax) in enumerate(zip(images, axes)):
        if idx < num_images:
            if net is not None:
                label = idx
                print(label)
            ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
            ax.axis('off')
            ax.text(0.5, -0.08, f'', transform=ax.transAxes, ha='center', fontsize=10)
        else:
            ax.axis('off')
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def load_diffusion(path, device):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    ema_dict = ld['ema']
    unet_dict = ld['unet']
    config = ld['config']
    config = DictConfig(config)
    unet = Unet(
        dim=config.unet.dim,
        image_size=config.image_size,
        dim_multiply=tuple(map(int, config.unet.dim_mults[1:-1].split(', '))),
        dropout=config.unet.dropout,
        device=device
    )
    unet.load_state_dict(unet_dict)
    diffusion = SanDiffusion(unet, config.diffusion.timesteps, device, sample_step=config.diffusion.sampling_timesteps)
    diffusion.ema.load_state_dict(ema_dict)
    return diffusion


def eval_diffusion(path, device):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    ema_dict = ld['ema']
    unet_dict = ld['unet']
    config = ld['config']
    config = DictConfig(config)
    eps_model = Unet(
        dim=config.unet.dim,
        image_size=config.image_size,
        dim_multiply=tuple(map(int, config.unet.dim_mults[1:-1].split(', '))),
        dropout=config.unet.dropout,
        device=device
    )
    eps_model.load_state_dict(unet_dict)
    diffusion = SanDiffusion(eps_model, config.diffusion.timesteps, device,
                             sample_step=config.diffusion.sampling_timesteps)
    diffusion.ema.load_state_dict(ema_dict)
    gen_sample(diffusion, 50000, f'{path}/fid')
    all_path = f'../dataset/dataset-{config.dataset_name}-all'
    fid = calculate_fid_given_paths([all_path, f'{path}/fid'], 128, "cuda:0", 2048, 8)
    print(fid)


@torch.inference_mode()
def show_sanitization(path, t, loop, device):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    config = DictConfig(ld['config'])
    config.sample_type = 'ddpm'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((config.image_size, config.image_size))
    ])
    tensor_list = get_dataset(config.dataset_name, transform)
    b = 9
    base = random.randint(0, 1000)
    tensors = tensor_list[base:base + b]
    tensors = torch.stack(tensors, dim=0)
    tensors = tensors.to(device)
    if config.attack == 'blended':
        trigger = transform(
            PIL.Image.open('../resource/blended/hello_kitty.jpeg')
        )
        trigger = trigger.unsqueeze(0).expand(b, -1, -1, -1)
        trigger = trigger.to(device)
        tensors = 0.8 * tensors + 0.2 * trigger.unsqueeze(0).expand(b, -1, -1, -1)
    elif config.attack == 'badnet':
        mask = PIL.Image.open(
            f'../resource/badnet/mask_{config.image_size}_{int(config.image_size / 10)}.png')
        mask = transform(mask)
        trigger = PIL.Image.open(
            f'../resource/badnet/trigger_{config.image_size}_{int(config.image_size / 10)}.png')
        trigger = transform(trigger)
        mask = mask.unsqueeze(0).expand(b, -1, -1, -1)
        trigger = trigger.unsqueeze(0).expand(b, -1, -1, -1)
        mask = mask.to(device)
        trigger = trigger.to(device)
        tensors = tensors * (1 - mask) + trigger
    x_0 = tensors
    diffusion = load_diffusion(path, device)
    san_list = [x_0]
    t_ = torch.tensor([t], device=device)
    # sanitization process
    for i in tqdm(range(loop), desc="iterate", total=loop, leave=False):
        # forward
        x_t = diffusion.q_sample(x_0, t_)
        san_list.append(x_t)
        # reverse
        for j in reversed(range(0, t)):
            x_t_m_1 = diffusion.p_sample(x_t, torch.tensor([j], device=device))
            x_t = x_t_m_1
        x_0 = x_t
        san_list.append(x_0)
    chain = torch.stack(san_list, dim=0)
    res = []
    for i in range(len(chain)):
        tensors = chain[i]
        torchvision.utils.save_image(tensors, f'{path}/res_{i}.png', nrow=int(math.sqrt(tensors.shape[0])))
        grid = make_grid(tensors, nrow=int(math.sqrt(tensors.shape[0])))
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        res.append(torchvision.transforms.transforms.ToTensor()(im))

    res = torch.stack(res, dim=0)
    plot_images(images=res, num_images=res.shape[0])


def get_args():
    parser = argparse.ArgumentParser(description='This script does amazing things.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--path', type=str)
    parser.add_argument('--mode', type=str, default='v')
    # parser.add_help = True
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = args.device
    path = args.path
    mode = args.mode
    if mode == 'fid':
        eval_diffusion(path, device)
    show_sanitization(path, 200, 8, device)
