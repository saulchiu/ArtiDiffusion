import math
import random
import PIL.Image
import torch
from denoising_diffusion_pytorch import Unet, Trainer
from PIL import Image
import sys
from torchvision.utils import make_grid
from omegaconf import DictConfig
import hydra

sys.path.append('../')
from backdoor_diffusion.badnet_diffusion_pred_noice import BadDiffusion, BadTrainer
import torchvision.transforms
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset, GaussianDiffusion
import matplotlib.pyplot as plt
import torch.utils
from tools.img import cal_ssim
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


def data_sanitization(diffusion, x_start, t, device='cuda:0'):
    x_t = diffusion.q_sample(x_start=x_start, t=torch.tensor([t], device=device))
    x_t_before = x_t
    for i in reversed(range(t)):
        pred_img, _ = diffusion.p_sample(x=x_t, t=i)
        x_t = pred_img
    x_start = x_t
    x_t = x_t_before
    return x_t, x_start


def iter_data_sanitization(diffusion, x_start, t=200, loop=8):
    tensor_list = [x_start]
    for i in range(loop):
        x_t, x_start = data_sanitization(diffusion, x_start, t)
        # whether append the sample with noise
        tensor_list.append(x_t)
        # append the sample after data sanitization
        tensor_list.append(x_start)
    tensors = torch.stack(tensor_list, dim=0)
    # plot_images(images=tensors, num_images=tensors.shape[0])
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
            save_and_sample_every=trainer_cfg.save_and_sample_every if trainer_cfg.save_and_sample_every > 0 else trainer_cfg.train_num_steps,
        )
    else:
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
    x_start_list = []
    for i in range(9):
        index = random.Random().randint(a=1, b=1000)
        print(index)
        x_start = transform(Image.open(f'../dataset/dataset-{cfg.dataset_name}-all/all_{index}.png'))
        x_start = x_start.to(device)
        x_start_list.append(x_start)
    if x_start.shape[1] != cfg.diffusion.image_size:
        prepare_bad_data(cfg)
    return diffusion, trainer, trigger, x_start_list

@hydra.main(version_base=None, config_path='../config/eval/', config_name='default')
def eval_result(cfg: DictConfig):
    t = cfg.step
    loop = cfg.loop
    path = cfg.path
    device = 'cuda:0'
    # load resnet
    ld = torch.load(f'{path}/result.pth', map_location=device)
    # draw_loss(ld, 300000, 700000)
    cfg = DictConfig(ld['config'])
    diff_cfg = cfg.diffusion
    # dataset_cfg = cfg.dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    diffusion, trainer, trigger, x_start_list = load_result(cfg, device)
    diffusion.load_state_dict(ld['diffusion'])
    diffusion = diffusion.to(device)
    tmp = []
    if cfg.attack == 'blended':
        trigger = transform(
            PIL.Image.open('../resource/blended/hello_kitty.jpeg')
        )
        trigger = trigger.to(device)
        for x_start in x_start_list:
            x_start = 0.8 * x_start + 0.2 * trigger
            tmp.append(x_start)
        x_start_list = tmp
    elif cfg.attack == 'badnet':
        mask = PIL.Image.open(f'../resource/badnet/mask_{diff_cfg.image_size}_{int(diff_cfg.image_size / 10)}.png')
        mask = transform(mask)
        mask = mask.to(device)
        trigger = PIL.Image.open(f'../resource/badnet/trigger_{diff_cfg.image_size}_{int(diff_cfg.image_size / 10)}.png')
        trigger = transform(trigger)
        trigger = trigger.to(device)
        for x_start in x_start_list:
            x_start = (1 - mask) * x_start + mask * trigger
            tmp.append(x_start)
        x_start_list = tmp
    elif cfg.attack == "benign":
        trigger = Image.open('../resource/blended/hello_kitty.jpeg')
        trigger = transform(trigger)
        trigger = trigger.to(device)
        # for x_start in x_start_list:
        #     x_start = 0.8 * x_start + 0.2 * trigger
        #     tmp.append(x_start)
        # x_start_list = tmp
    x_starts = torch.stack(x_start_list, dim=0)
    chain = iter_data_sanitization(diffusion, x_starts, t, loop)
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





if __name__ == '__main__':
    eval_result()
