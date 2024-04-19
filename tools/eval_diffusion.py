import math

import numpy as np
import torch
from denoising_diffusion_pytorch import Unet, Trainer
from PIL import Image
import sys

sys.path.append('../')
from backdoor_diffusion.badnet_diffusion import BadDiffusion
import torchvision.transforms
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset, GaussianDiffusion
from torch.utils.data import DataLoader
from tools.img import save_one
import matplotlib.pyplot as plt
from torchvision import utils
import PIL.Image
import torch.utils


def plot_images(images, num_images):
    """
    Plot a list of images in a grid.

    Parameters:
    - images: A torch tensor of shape (num_images, channels, height, width)
    - num_images: The number of images to plot
    """
    # Calculate the number of rows and columns
    cols = int(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # Flatten the array for easier iteration

    # Plot each image on the grid
    for idx, (img, ax) in enumerate(zip(images, axes)):
        if idx < num_images:  # Only plot the actual number of images
            ax.imshow(img.permute(1, 2, 0).cpu().numpy())
            ax.axis('off')
        else:
            ax.axis('off')  # Turn off the last empty subplot

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def load_bad_diffusion(path):
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


def load_diffusion(path, device='cuda:0'):
    ld = torch.load(path)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )
    diffusion.load_state_dict(ld['model'])
    diffusion = diffusion.to(device=device)
    return diffusion


def sample_from_diffusion(diff_path, batch=4):
    with torch.inference_mode():
        diffusion = load_bad_diffusion(diff_path)
        sampled_images = diffusion.sample(batch_size=batch)
        # utils.save_image(sampled_images, fp='./test.png', nrow=4)
        plot_images(sampled_images, batch)


def sample_and_reconstruct(diffusion, x_start, t=10, device='cuda:0', plot=False):
    tensor_list = [x_start]
    x_noice = diffusion.q_sample(x_start=x_start, t=torch.tensor([t]).to(device))
    tensor_list.append(x_noice)
    x_noice = x_noice.reshape(1, 3, 32, 32)
    _, x_s = diffusion.p_sample(x=x_noice, t=t)
    x_s = x_s.reshape(3, 32, 32)
    tensor_list.append(x_s)
    tensors = torch.stack(tensor_list, dim=0)
    if plot:
        plot_images(tensors, len(tensor_list))
    return x_s


def sample_and_reconstruct_loop(diffusion, x_start, t=10, device='cuda:0', plot=False, loop=5):
    tensor_list = [x_start]
    for i in range(loop):
        x_t1 = sample_and_reconstruct(diffusion, x_start, t)
        tensor_list.append(x_t1)
        x_start = x_t1
    tensors = torch.stack(tensor_list, dim=0)
    plot_images(tensors, tensors.shape[0])


if __name__ == '__main__':
    device = 'cuda:0'
    t = 5
    loop = 15
    diffusion = load_diffusion('../backdoor_diffusion/res_badnet_grid_cifar10_step10k_ratio2/model-10.pt', device=device)
    x_start = Image.open('../dataset/dataset-cifar10-badnet-trigger_image_grid/bad_0.png')
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32))
    ])
    x_start = trans(x_start)
    x_start = x_start.to(device)
    sample_and_reconstruct_loop(diffusion, x_start, t, device, False, loop)
