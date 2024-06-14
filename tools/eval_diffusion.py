import argparse
import ast
import math
import random

import PIL.Image
import detectors
import timm
import torch
from denoising_diffusion_pytorch import Unet, Trainer
from PIL import Image
import sys
import torchvision.transforms.transforms as T
from torchvision.transforms import transforms

sys.path.append('../')
from backdoor_diffusion.badnet_diffusion_pred_noice import BadDiffusion, BadTrainer
import torchvision.transforms
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset, GaussianDiffusion
import matplotlib.pyplot as plt
import torch.utils
from tools.img import cal_ssim
from models.resnet import ResNet18, ResNet50
from tools.classfication import MyLightningModule
from torch.utils.data import DataLoader
from tools.dataset import transform_cifar10
from backdoor_diffusion.benign_deffusion import BenignTrainer
from tools.prepare_data import prepare_bad_data
from tools.time import now


def plot_images(images, num_images, net=None):
    label = ''
    indexes = []
    if net is not None:
        y_p = net(images)
        _, indexes = y_p.max(1)
    cols = int(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    figsize_width = cols * 5
    figsize_height = rows * 5
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height))
    axes = axes.flatten()  # Flatten the array for easier iteration
    # Plot each image on the grid
    for idx, (img, ax) in enumerate(zip(images, axes)):
        if idx < num_images:  # Only plot the actual number of images
            img_ssim = cal_ssim(img, images[0])
            if net is not None:
                label = idx
                print(label)
            ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
            ax.axis('off')
            ax.text(0.5, -0.08, f'SSIM: {img_ssim:.2f}, {label}', transform=ax.transAxes, ha='center',
                    fontsize=10)
        else:
            ax.axis('off')  # Turn off the last empty subplot

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def data_sanitization(diffusion, x_start, t=10, device='cuda:0', plot=False):
    x_t = diffusion.q_sample(x_start=x_start, t=torch.tensor([t]).to(device))
    x_T = x_t
    x_t = x_t.unsqueeze(0)
    for i in reversed(range(t)):
        pred_img, _ = diffusion.p_sample(x=x_t, t=i + 1)
        x_t = pred_img
    x_z = x_t.squeeze(0)
    return x_T, x_z


def iter_data_sanitization(diffusion, x_start, t=200, loop=8):
    tensor_list = [x_start]
    for i in range(loop):
        x_t, x_z = data_sanitization(diffusion, x_start, t)
        tensor_list.append(x_t)
        tensor_list.append(x_z)
        x_start = x_z
    tensors = torch.stack(tensor_list, dim=0)
    plot_images(images=tensors, num_images=tensors.shape[0])
    return tensor_list


def load_result(cfg, device):
    diffusion, trainer, trigger, x_start = None, None, None, None
    unet_cfg = cfg.noise_predictor
    diff_cfg = cfg.diffusion
    trainer_cfg = cfg.trainer
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    model = Unet(
        dim=unet_cfg.dim,
        dim_mults=tuple(map(int, unet_cfg.dim_mults[1:-1].split(', '))),
        flash_attn=unet_cfg.flash_attn
    )
    model = model.to(device)
    if cfg.attack == "benign":
        diffusion = GaussianDiffusion(
            model,
            image_size=diff_cfg.image_size,
            timesteps=diff_cfg.timesteps,  # number of steps
            sampling_timesteps=diff_cfg.sampling_timesteps,
            objective=diff_cfg.objective
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
    else:
        trigger_path = diff_cfg.trigger
        trigger = Image.open(trigger_path)
        trigger = transform(trigger).to(device)
        diffusion = BadDiffusion(
            model,
            image_size=diff_cfg.image_size,
            timesteps=diff_cfg.timesteps,  # number of steps
            sampling_timesteps=diff_cfg.sampling_timesteps,
            objective=diff_cfg.objective,
            trigger=trigger,
            device=device,
            attack=diff_cfg.attack,
            gamma=0
        )
    index = random.Random().randint(a=1, b=1000)
    # index = 25
    x_start = transform(Image.open(f'../dataset/dataset-{cfg.dataset_name}-all/all_{index}.png'))
    if x_start.shape[1] != cfg.diffusion.image_size:
        prepare_bad_data(cfg)
    x_start = x_start.to(device)
    return diffusion, trainer, trigger, x_start


def draw_loss(result, start, end):
    # 提取损失列表
    loss_list = result['loss_list']

    # 根据start和end索引进行切片
    loss_list_sliced = loss_list[start:end]  # Python切片包括开始索引，不包括结束索引

    # 提取不同模式的损失列表
    losses_mode_0 = [item['loss'] for item in loss_list_sliced if item['mode'] == 0]
    losses_mode_1 = [item['loss'] for item in loss_list_sliced if item['mode'] == 1]
    losses_all = [item['loss'] for item in loss_list_sliced]  # 所有模式的损失

    # 创建一个图形窗口，并设置子图的布局为1行3列
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1行3列

    # 绘制mode为0的损失曲线（蓝色）
    axs[0].plot(losses_mode_0, color='blue')
    axs[0].set_title('Benign Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # 绘制mode为1的损失曲线（红色）
    axs[1].plot(losses_mode_1, color='red')
    axs[1].set_title('Poisoning Loss')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)

    # 绘制所有模式的损失曲线（黑色虚线）
    axs[2].plot(losses_all, linestyle='--', color='black')
    axs[2].set_title('All Modes Loss')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True)

    # 调整子图间距
    plt.tight_layout()

    # 显示图表
    plt.show()


def eval_tmp(path, attack, x_start, device='cuda:0'):
    ld = torch.load(path)
    unet = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    unet.load_state_dict(ld['unet'])
    diffusion = BadDiffusion(
        model=unet,
        image_size=32,
        sampling_timesteps=250,
        objective='pred_noise',
        trigger=None,
        device='cuda:0',
        attack='blended',
        gamma=1e-3,
        timesteps=1000
    )
    diffusion.load_state_dict(ld['diffusion'])
    if attack == 'blended':
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize((64, 64))
        ])
        trigger = transform(
            PIL.Image.open('../resource/blended/hello_kitty.jpeg')
        )
        trigger = trigger.to(device)
        x_start = 0.8 * x_start + 0.2 * trigger
    iter_data_sanitization(diffusion, x_start, 8, 200)


from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(version_base=None, config_path='../config/eval/', config_name='default')
def eval_result(cfg: DictConfig):
    t = cfg.step
    loop = cfg.loop
    path = cfg.path
    do_sample = cfg.sample
    device = 'cuda:0'
    # load resnet
    ld = torch.load(f'{path}/result.pth', map_location=device)
    # draw_loss(ld, 300000, 700000)
    cfg = DictConfig(ld['config'])
    # fid_list = ld['fid_list']
    diff_cfg = cfg.diffusion
    # dataset_cfg = cfg.dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    diffusion, trainer, trigger, x_start = load_result(cfg, device)
    diffusion.load_state_dict(ld['diffusion'])
    diffusion = diffusion.to(device)
    name = cfg.dataset_name
    if cfg.attack == 'blended':
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize((32, 32))
        ])
        trigger = transform(
            PIL.Image.open('../resource/blended/hello_kitty.jpeg')
        )
        trigger = trigger.to(device)
        x_start = 0.8 * x_start + 0.2 * trigger
    elif cfg.attack == 'badnet':
        mask = PIL.Image.open(f'../resource/badnet/mask_{diff_cfg.image_size}_{int(diff_cfg.image_size / 10)}.png')
        mask = transform(mask)
        mask = mask.to(device)
        x_start = (1 - mask) * x_start + mask * trigger
    elif cfg.attack == "benign":
        trigger = Image.open('../resource/blended/hello_kitty.jpeg')
        trigger = transform(trigger)
        trigger = trigger.to(device)
        # x_start = 0.8 * x_start + 0.2 * trigger
        x_start = x_start
    iter_data_sanitization(diffusion, x_start, t, loop)
