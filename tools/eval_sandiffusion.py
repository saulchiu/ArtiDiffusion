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
from tools.dpm_solver import DPM_Solver, NoiseScheduleVP


def expand_dims(v, dims):
    return v[(...,) + (None,)*(dims - 1)]

def model_wrapper(
        model,
        noise_schedule,
        model_type="noise",
        model_kwargs={},
        guidance_type="uncond",
        condition=None,
        unconditional_condition=None,
        guidance_scale=1.,
        classifier_fn=None,
        classifier_kwargs={},
):
    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


def gen_sample(diffusion, total_sample, target_folder, sampler: str):
    rm_if_exist(target_folder)
    loop = int(total_sample / 64)
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
        sample_fn = lambda batch: dpm.sample(x=torch.randn(batch, 3, 32, 32, device="cuda:0"))
    else:
        raise NotImplementedError
    for _ in tqdm(range(loop)):
        fake_sample = sample_fn(64)
        save_tensor_images(fake_sample, target_folder)
    if (total_sample - loop * 64) != 0:
        fake_sample = sample_fn(total_sample - loop * 64)
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


def gen_and_cal_fid(path, device, sampler):
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
    gen_sample(diffusion, 50000, f'{path}/fid', sampler)
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
    parser.add_argument('--mode', type=str, default='no_fid')
    parser.add_argument("--t", type=int, default=200)
    parser.add_argument("--l", type=int, default=8)
    parser.add_argument("--sampler", type=str, default="ddim")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = args.device
    path = args.path
    mode = args.mode
    timestep = args.t
    loop = args.l
    sampler = args.sampler
    if mode == 'fid':
        gen_and_cal_fid(path, device, sampler)
    show_sanitization(path, timestep, loop, device)
