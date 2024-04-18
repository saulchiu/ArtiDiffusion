import numpy as np
import torch
from denoising_diffusion_pytorch import Unet, Trainer
from PIL import Image
import sys

sys.path.append('../')
from backdoor_diffusion.badnet_diffusion import BadDiffusion
import torchvision.transforms
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset
from torch.utils.data import DataLoader
from tools.img import save
import matplotlib.pyplot as plt


def plt_img(tensor, batch):
    sampled_images = tensor.cpu().numpy()
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(sampled_images):
        plt.subplot(1, batch, i + 1)
        plt.imshow(image.transpose(1, 2, 0))
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.show()


def load_diffusion(path):
    ld = torch.load(path)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    diffusion = BadDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        objective='pred_x0',
        trigger=None
    )
    diffusion.load_state_dict(ld['model'])
    diffusion = diffusion.to('cuda:0')
    return diffusion


if __name__ == '__main__':
    path = '../backdoor_diffusion/res_badnet_grid_cifar10_step1k_ratio2/model-1.pt'
    diffusion = load_diffusion(path)
    sampled_images = diffusion.sample(batch_size=16)
    plt_img(tensor=sampled_images, batch=16)
