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

from tools.time import now

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
            factor_list=ast.literal_eval(str(diff_cfg.factor_list)),
            device=device,
            reverse_step=diff_cfg.reverse_step,
            attack=diff_cfg.attack,
            gamma=0
        )
        trainer = BadTrainer(
            diffusion,
            bad_folder=trainer_cfg.bad_folder,
            good_folder=trainer_cfg.good_folder,
            train_batch_size=trainer_cfg.train_batch_size,
            train_lr=trainer_cfg.train_lr,
            train_num_steps=trainer_cfg.train_num_steps,
            gradient_accumulate_every=trainer_cfg.gradient_accumulate_every,
            ema_decay=trainer_cfg.ema_decay,
            amp=trainer_cfg.amp,
            calculate_fid=trainer_cfg.calculate_fid,
            ratio=trainer_cfg.ratio,
            results_folder=trainer_cfg.results_folder,
            server=trainer_cfg.server,
            save_and_sample_every=trainer_cfg.save_and_sample_every if trainer_cfg.save_and_sample_every > 0 else trainer_cfg.train_num_steps,
        )
    index = random.Random().randint(a=1, b=1000)
    # index = 25
    x_start = transform(Image.open(f'{trainer_cfg.good_folder}/good_{index}.png'))
    if x_start.shape[1] != cfg.diffusion.image_size:
        prepare_bad_data(cfg)
    x_start = x_start.to(device)
    return diffusion, trainer, trigger, x_start


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
    cfg = DictConfig(ld['config'])
    fid_list = ld['fid_list']
    diff_cfg = cfg.diffusion
    dataset_cfg = cfg.dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    diffusion, trainer, trigger, x_start = load_result(cfg, device)
    # ld = torch.load('../results/blended/imagenette/exp2/result.pth')
    diffusion.load_state_dict(ld['diffusion'])
    diffusion = diffusion.to(device)
    # trainer.my_sample()
    if do_sample:
        batches = [16, 9]
        diffusion.is_ddim_sampling = False
        all_images_list = list(map(lambda n: diffusion.sample(batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        torchvision.utils.save_image(all_images, f'{path}/sample-{now()}.png',
                                     nrow=int(math.sqrt(trainer.num_samples)))
    name = cfg.dataset_name
    if cfg.attack == 'blended':
        print()
        # x_start = 0.8 * x_start + 0.2 * trigger
    elif cfg.attack == 'badnet':
        mask = PIL.Image.open('../resource/badnet/mask_32_3.png')
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
    # if cal_fid:
    #     fid = trainer.fid_scorer.fid_score()
    #     fid_list.append(fid)
    #     print(fid_list)
    #     ld['fid_list'] = fid_list
    #     torch.save(ld, path)


if __name__ == '__main__':
    eval_result()
