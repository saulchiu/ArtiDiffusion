import argparse
import math
import random

import PIL
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
from omegaconf import DictConfig
from torchvision.transforms.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F

import sys

from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append('../')
from diffusion.sandiffusion import SanDiffusion
from tools.dataset import save_tensor_images, rm_if_exist, load_dataloader
from tools.prepare_data import get_dataset
from diffusion.unet import Unet
from diffusion.dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper


def get_sample_fn(diffusion, sampler, sample_step):
    if sampler == "ddpm":
        sample_fn = diffusion.ddpm_sample
    elif sampler == "ddim":
        sample_fn = diffusion.ddim_sample
    elif sampler == "dpm_solver":
        ns = NoiseScheduleVP(schedule='discrete', alphas_cumprod=diffusion.alpha_bar)
        model_fn_continuous = model_wrapper(
            diffusion.ema.ema_model,
            ns,
            model_type="noise",
            model_kwargs={},
            guidance_type="uncond",
            condition=None,
            guidance_scale=0.0,
            classifier_fn=None,
            classifier_kwargs={},
        )
        dpm = DPM_Solver(model_fn_continuous, ns, algorithm_type='dpmsolver', correcting_x0_fn=None)
        sample_fn = lambda batch: dpm.sample(x=torch.randn(
            batch, diffusion.eps_model.channel, diffusion.image_size, diffusion.image_size,
            device=diffusion.eps_model.device,
        ), steps=sample_step, order=2)
    else:
        raise NotImplementedError
    return sample_fn


def gen_sample(diffusion, total_sample, target_folder, sampler, sample_step, batch):
    rm_if_exist(target_folder)
    loop = int(total_sample / batch)
    sample_fn = get_sample_fn(diffusion, sampler, sample_step)
    for _ in tqdm(range(loop)):
        fake_sample = sample_fn(batch)
        save_tensor_images(fake_sample, target_folder)
    if (total_sample - loop * batch) != 0:
        fake_sample = sample_fn(total_sample - loop * batch)
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
    diffusion = SanDiffusion(
        unet,
        config.diffusion.timesteps,
        device,
        sample_step=config.diffusion.sampling_timesteps,
        beta_schedule=config.diffusion.beta_schedule
    )
    diffusion.ema.load_state_dict(ema_dict)
    return diffusion


def gen_and_cal_fid(path, device, sampler, sample_step, gen_batch):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    ema_dict = ld['ema']
    unet_dict = ld['unet']
    config = ld['config']
    config = DictConfig(config)
    print(config)
    eps_model = Unet(
        dim=config.unet.dim,
        image_size=config.image_size,
        dim_multiply=tuple(map(int, config.unet.dim_mults[1:-1].split(', '))),
        dropout=config.unet.dropout,
        device=device
    )
    eps_model.load_state_dict(unet_dict)
    diffusion = SanDiffusion(eps_model, config.diffusion.timesteps, device,
                             sample_step=config.diffusion.sampling_timesteps,
                             beta_schedule=config.diffusion.beta_schedule,
                             )
    diffusion.ema.load_state_dict(ema_dict)
    gen_sample(diffusion, 50000, f'{path}/fid', sampler, sample_step=sample_step, batch=gen_batch)
    all_path = f'../dataset/dataset-{config.dataset_name}-all'
    fid = calculate_fid_given_paths([all_path, f'{path}/fid'], 128, "cuda:0", 2048, 8)
    print(fid)


def cal_mse(path, device, num, batch):
    diffusion = load_diffusion(path, device)
    loop = int(num / batch)
    config = torch.load(f'{path}/result.pth', map_location=device)['config']
    config = DictConfig(config)
    all_path = f'../dataset/dataset-{config.dataset_name}-all'
    trans = Compose([
        ToTensor(), Resize((config.image_size, config.image_size))
    ])
    all_loader = load_dataloader(path=all_path, trans=trans, batch=config.batch)
    loss_fn = F.mse_loss
    total_loss = torch.zeros(size=(), device=device)
    c_loop = 0
    with tqdm(initial=c_loop, total=loop) as pbar:
        for _ in range(0, loop):
            x_0 = next(all_loader)
            x_0 = x_0.to(device)
            eps = torch.randn_like(x_0, device=device)
            t = torch.randint(low=0, high=1000, size=(x_0.shape[0],), device=device).long()
            x_t = diffusion.q_sample(x0=x_0, t=t, eps=eps)
            eps_theta = diffusion.ema.ema_model(x_t, t)
            loss = loss_fn(eps_theta, eps)
            total_loss += loss
            pbar.set_description(f'c_loss: {loss:.4f}')
            c_loop += 1
            pbar.update(1)
        total_loss = total_loss / loop
    print(f'total loss of {num} samples: {total_loss: .5f}')


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
    b = 16
    base = random.randint(0, 1000)
    tensors = tensor_list[base:base + b]
    tensors = torch.stack(tensors, dim=0)
    tensors = tensors.to(device)
    if config.attack == 'blended':
        trigger = transform(
            PIL.Image.open('../resource/blended/hello_kitty.jpeg')
        )
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
        # san_list.append(x_t)
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
    class ModeAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            if values == 'sanitization':
                print("\ndevice, path of result.pth, diffuse and reverse timestep t, iteration l")
            elif values == 'fid':
                print("\nrequire: device; path; sampler name, e.g. ddim; sampler batch; sample_step.")
            elif values == 'mse':
                print("\nrequire: device, path, batch, total sample num")

    """
    the describe of mode
        sanitization: show sanitization
            require: device, path of result.pth, diffuse and reverse timestep t, iteration l
        fid: calculate fid of diffusion
            require: device; path; sampler name, e.g. ddim; sampler batch; sample_step.
        mse: calculate mse of diffusion
            require: device, path, batch, total sample num
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sanitization', action=ModeAction)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--path', type=str)
    parser.add_argument("--batch", type=int, default=64)

    # sanitization parameter
    parser.add_argument("--t", type=int, default=200)
    parser.add_argument("--l", type=int, default=8)

    # fid parameter
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--sample_step", type=int, default=1000)

    # mse parameter
    parser.add_argument("--num", type=int, default=1e4)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    mode = args.mode
    if mode == 'sanitization':
        device = args.device
        path = args.path
        timestep = args.t
        loop = args.l
        show_sanitization(path, timestep, loop, device)
    elif mode == 'fid':
        device = args.device
        path = args.path
        sampler = args.sampler
        batch = args.batch
        sample_step = args.sample_step
        gen_and_cal_fid(path, device, sampler, gen_batch=batch, sample_step=sample_step)
    elif mode == 'mse':
        path = args.path
        device = args.device
        batch = args.batch
        num = args.num
        cal_mse(path, device, num, batch)
