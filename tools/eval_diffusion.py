import ast
import math

import PIL.Image
import detectors
import timm
import torch
from denoising_diffusion_pytorch import Unet, Trainer
from PIL import Image
import sys
import torchvision.transforms.transforms as T


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

def sample_and_reconstruct(diffusion, x_start, t=10, device='cuda:0', plot=False):
    x_t = diffusion.q_sample(x_start=x_start, t=torch.tensor([t]).to(device))
    x_t = x_t.unsqueeze(0)
    for i in reversed(range(t)):
        pred_img, _ = diffusion.p_sample(x=x_t, t=i + 1)
        x_t = pred_img
    x_t = x_t.squeeze(0)
    return x_t


def sample_and_reconstruct_loop(diffusion, x_start, t=10, loop=5):
    tensor_list = [x_start]
    for i in range(loop):
        x_t1 = sample_and_reconstruct(diffusion, x_start, t)
        tensor_list.append(x_t1)
        x_start = x_t1
    tensors = torch.stack(tensor_list, dim=0)
    plot_images(images=tensors, num_images=tensors.shape[0])
    return tensor_list


def load_model(cfg, device):
    diffusion, trainer, trigger = None, None, None
    unet_cfg = cfg.noise_predictor
    diff_cfg = cfg.diffusion
    trainer_cfg = cfg.trainer
    model = Unet(
        dim=unet_cfg.dim,
        dim_mults=tuple(map(int, unet_cfg.dim_mults[1:-1].split(', '))),
        flash_attn=unet_cfg.flash_attn
    )
    model = model.to(device)
    trigger_path = diff_cfg.trigger
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])

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
            attack=diff_cfg.attack
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

    return diffusion, trainer, trigger


from omegaconf import OmegaConf, DictConfig


def eval_result(t, loop, path, cal_fid=False):
    device = 'cuda:0'
    # load resnet
    ld = torch.load(path, map_location=device)
    cfg = DictConfig(ld['config'])
    unet_cfg = cfg.noise_predictor
    diff_cfg = cfg.diffusion
    trigger_path = diff_cfg.trigger
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    diffusion, trainer, trigger = load_model(cfg, device)
    # ld = torch.load('../results/blended/imagenette/exp2/result.pth')
    diffusion.load_state_dict(ld['diffusion'])
    diffusion = diffusion.to(device)
    name = cfg.dataset_name
    normal_data = None
    if name == 'cifar10':
        normal_data = torchvision.datasets.CIFAR10(
            root='../data/', train=True, transform=transform, download=False
        )
    elif name == 'imagenette':
        normal_data = torchvision.datasets.Imagenette(
            root='../data/', split='train', size='full', download=False, transform=transform
        )
    elif name == 'gtsrb':
        normal_data = torchvision.datasets.GTSRB(
            root='../data', transform=transform
        )
    normal_loader = torch.utils.data.DataLoader(dataset=normal_data, batch_size=128, shuffle=True, num_workers=1)
    x_start, index = next(iter(normal_loader))
    x_start = x_start[1]
    index = index[1]
    x_start = x_start.squeeze(0)
    x_start = x_start.to(device)
    if cfg.attack == 'blended':
        print()
        # x_start = 0.8 * x_start + 0.2 * trigger
    elif cfg.attack == 'badnet':
        mask = PIL.Image.open('../resource/badnet/trigger_image.png')
        mask = transform(mask)
        mask = mask.to(device)
        x_start = (1 - mask) * x_start + mask * trigger
    elif cfg.attack == "benign":
        trigger = Image.open('../resource/blended/hello_kitty.jpeg')
        trigger = transform(trigger)
        trigger = trigger.to(device)
        # x_start = 0.8 * x_start + 0.2 * trigger
        x_start = x_start
    print(f'real label is: {index}')
    sample_and_reconstruct_loop(diffusion, x_start, t, loop)
    if cal_fid:
        fid = trainer.fid_scorer.fid_score()
        print(fid)


if __name__ == '__main__':
    eval_result(200, 8, '../results/blended/imagenette/202405260104_100k/result.pth')
