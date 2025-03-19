import argparse
import math
import os
import random

import PIL
import PIL.Image
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import gaussian_filter
from torchvision.transforms.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity

import sys

from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append('../')
from diffusion.diffusion_model import DiffusionModel
from tools.dataset import save_tensor_images, rm_if_exist, load_dataloader
from tools.prepare_data import get_dataset, tensor2bad
from diffusion.unet import Unet
from diffusion.dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper
from defence.sample import infer_clip_p_sample
from defence.anp.anp_defence import convert_model
from defence.sample import anp_sample, infer_clip_p_sample
from tools.ftrojann_transform import get_ftrojan_transform
from tools.ctrl_transform import ctrl
from tools.utils import unsqueeze_expand
from tools.time import now
from tools.img import rgb_tensor_to_lab_tensor, split_lab_channels, lab_tensor_to_rgb_tensor


def get_sample_fn(diffusion, sampler, sample_step):
    if sampler == "ddpm":
        # sample_fn = diffusion.ddpm_sample
        sample_fn = lambda batch: diffusion.ddim_sample(batch, sampling_timesteps=sample_step)
    elif sampler == "ddim":
        # sample_fn = diffusion.ddim_sample
        sample_fn = lambda batch: diffusion.ddim_sample(batch, sampling_timesteps=sample_step)
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
        save_tensor_images(fake_sample.cpu(), target_folder)
    if (total_sample - loop * batch) != 0:
        fake_sample = sample_fn(total_sample - loop * batch)
        save_tensor_images(fake_sample.cpu(), target_folder)
    return


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def plot_images(images: torch.tensor, num_images, net=None):
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
    plt.savefig('./test.jpg')
    plt.show()


def load_diffusion(path, device) -> DiffusionModel:
    ld = torch.load(f'{path}/result.pth', map_location=device)
    ema_dict = ld['ema']
    unet_dict = ld['unet']
    # config = ld['config']
    # config = DictConfig(config)
    config = OmegaConf.load(f'{path}/config.yaml')
    config = DictConfig(config)

    # test different beta schedule
    # config.diffusion.beta_schedule = 'scaled_linear'
    # config.diffusion.beta_schedule = 'squaredcos_cap_v2'
    # config.diffusion.beta_schedule = 'jsd'
    unet = Unet(
        dim=config.unet.dim,
        image_size=config.image_size,
        dim_multiply=tuple(map(int, config.unet.dim_mults[1:-1].split(', '))),
        dropout=config.unet.dropout,
        device=device
    )
    unet.load_state_dict(unet_dict)
    diffusion = DiffusionModel(
        unet,
        config.diffusion.timesteps,
        device,
        sample_step=config.diffusion.sampling_timesteps,
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    )
    diffusion.ema.load_state_dict(ema_dict)
    return diffusion


def gen_adv_sample(pth_path, device, adv_path):
    dm = load_diffusion(pth_path, device)
    backdoor_noise = torch.randn(size=(16, 3, 32, 32), device=device)
    transform = Compose([
        ToTensor(), Resize((32, 32))
    ])
    mask = PIL.Image.open(
        f'../resource/badnet/mask_32_3.png')
    mask = transform(mask)
    trigger = PIL.Image.open(
        f'../resource/badnet/trigger_32_3.png')
    trigger = transform(trigger)
    mask = mask.unsqueeze(0).expand(16, -1, -1, -1)
    trigger = trigger.unsqueeze(0).expand(16, -1, -1, -1)
    mask = mask.to(device)
    trigger = trigger.to(device)
    x_t = backdoor_noise * (1 - mask) + trigger
    x_t = dm.ddim_sample(16, x_t)
    save_tensor_images(x_t, adv_path)
    return x_t

def gen_and_cal_fid(path, device, sampler, sample_step, gen_batch, total):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    ema_dict = ld['ema']
    unet_dict = ld['unet']
    # config = ld['config']
    # config = DictConfig(config)
    config = OmegaConf.load(f'{path}/config.yaml')
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
    diffusion = DiffusionModel(eps_model, config.diffusion.timesteps, device,
                             sample_step=config.diffusion.sampling_timesteps,
                             beta_schedule=config.diffusion.beta_schedule,
                             beta_start=config.diffusion.beta_start,
                             beta_end=config.diffusion.beta_end
                             )
    diffusion.ema.load_state_dict(ema_dict)
    gen_sample(diffusion, total, f'{path}/fid', sampler, sample_step=sample_step, batch=gen_batch)
    all_path = f'../dataset/dataset-{config.dataset_name}-all'
    fid = calculate_fid_given_paths([all_path, f'{path}/fid'], 128, "cuda:0", 2048, 8)
    print(fid)
    filename = f'{path}/res.md'
    content_to_append = f'\n{now()}_{sampler}_{sample_step}: {fid}\n'
    with open(filename, 'a') as f:
        f.write(content_to_append)
    return fid


@torch.inference_mode()
def purification(path, t, loop, device, defence="None", batch=None, plot=True, target=False, fix_seed=False):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    # config = DictConfig(ld['config'])
    config = OmegaConf.load(f'{path}/config.yaml')
    config = DictConfig(config)
    config.sample_type = 'ddpm'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((config.image_size, config.image_size))
    ])
    tensor_list = get_dataset(config.dataset_name, transform, target)
    b = 16 if batch is None else batch
    base = random.randint(0, 10000) if fix_seed is False else 64
    tensors = tensor_list[base:base + b]
    tensors = torch.stack(tensors, dim=0)
    tensors = tensors.to(device)
    '''
    load benign model but use poisoning sample
    '''
    # config.attack = 'benign'
    # config.attack = 'badnet'
    # config.attack = 'blended'
    # config.attack = 'wanet'

    x_0 = tensor2bad(config, tensors, transform, device)
    diffusion = load_diffusion(path, device)
    san_list = [x_0]
    # eval defence here
    if defence == 'None':
        p_sample = diffusion.p_sample
    elif defence == 'anp':
        perturb_model = convert_model(diffusion.eps_model)
        perturb_model.load_state_dict(torch.load(f'{path}/result_anp.pth')['unet'])
        perturb_model.to(device)
        diffusion.eps_model = perturb_model
        p_sample = lambda x_t, t: anp_sample(diffusion=diffusion, xt=x_t, t=t)
    elif defence == 'rnp':
        perturb_model = convert_model(diffusion.eps_model)
        perturb_model.load_state_dict(torch.load(f'{path}/result_rnp.pth')['unet'])
        perturb_model.to(device)
        diffusion.eps_model = perturb_model
        p_sample = lambda x_t, t: anp_sample(diffusion=diffusion, xt=x_t, t=t)
    elif defence == "infer_clip":
        p_sample = lambda x_t, t: infer_clip_p_sample(diffusion, x_t, t + 1)
    else:
        raise NotImplementedError(defence)
    # sanitization process
    with tqdm(initial=0, total=loop) as pbar:
        for i in range(loop):
            # save img to collect data
            p = f'{path}/purify_{i}'
            rm_if_exist(p)
            os.makedirs(p, exist_ok=True)
            save_tensor_images(x_0, p)
            # forward
            x_t = diffusion.q_sample(x_0, torch.tensor([t], device=device))
            # reverse
            for j in reversed(range(0, t)):
                x_t_m_1 = p_sample(x_t, torch.tensor([j], device=device))
                x_t = x_t_m_1
            x_0 = x_t
            san_list.append(x_0)
            pbar.update(1)
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
    if plot:
        plot_images(images=res, num_images=res.shape[0])
    return tensors, res

@torch.inference_mode()
def inpainting(path, t, loop, device, defence="None", batch=None, plot=True, target=False, fix_seed=False):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    # config = DictConfig(ld['config'])
    config = OmegaConf.load(f'{path}/config.yaml')
    config = DictConfig(config)
    config.sample_type = 'ddpm'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((config.image_size, config.image_size))
    ])
    tensor_list = get_dataset(config.dataset_name, transform, target)
    b = 16 if batch is None else batch
    base = random.randint(0, 10000) if fix_seed is False else 64
    tensors = tensor_list[base:base + b]
    tensors = torch.stack(tensors, dim=0)
    tensors = tensors.to(device)
    '''
    load benign model but use poisoning sample
    '''
    # config.attack = 'benign'
    # config.attack = 'badnet'
    # config.attack = 'blended'
    # config.attack = 'wanet'
    x_0 = tensor2bad(config, tensors, transform, device)
    diffusion = load_diffusion(path, device)
    san_list = [x_0.cpu()]
    # mask x_0s
    mask_list = []
    for i in range(x_0.shape[0]):
        _, c, h, w = x_0.shape
        mask = torch.ones((c, h, w), device=device)
        step = int(config.image_size / 4)
        start = random.randint(0, step * 2)
        mask[:, start:(start + step), start:(start + step)] = 0
        mask_list.append(mask)
    mask = torch.stack(mask_list, dim=0)
    # eval defence here
    if defence == 'None':
        p_sample = diffusion.p_sample
    elif defence == 'anp':
        perturb_model = convert_model(diffusion.eps_model)
        perturb_model.load_state_dict(torch.load(f'{path}/result_anp.pth')['unet'])
        perturb_model.to(device)
        diffusion.eps_model = perturb_model
        p_sample = lambda x_t, t: anp_sample(diffusion=diffusion, xt=x_t, t=t)
    elif defence == 'rnp':
        perturb_model = convert_model(diffusion.eps_model)
        perturb_model.load_state_dict(torch.load(f'{path}/result_rnp.pth')['unet'])
        perturb_model.to(device)
        diffusion.eps_model = perturb_model
        p_sample = lambda x_t, t: anp_sample(diffusion=diffusion, xt=x_t, t=t)
    elif defence == "infer_clip":
        p_sample = lambda x_t, t: infer_clip_p_sample(diffusion, x_t, t + 1)
    else:
        raise NotImplementedError(defence)
    x_0 = x_0 * mask + (1 - mask) * torch.randn_like(x_0, device=x_0.device)
    # x_0 = x_0 * mask
    # x_0 = x_0 * mask + 1 * (1 - mask)
    san_list.append(x_0.cpu())
    x_t = x_0.clone()
    # t = 400
    # loop = 16
    distance = int(t / loop)
    decreasing_list = [t - i * distance for i in range(loop)]
    print(decreasing_list)
    for i in tqdm(range(0, loop)):
        x_t = diffusion.q_sample(x_t, (torch.ones(size=(x_0.shape[0],), device=device) * decreasing_list[i]).to(torch.int64))
        for j in reversed(range(0, decreasing_list[i])):
            x_t_m_1 = p_sample(x_t, torch.tensor([j], device=device))
            x_t = x_t_m_1
        x_t = x_0 * mask + (1 - mask) * x_t
        san_list.append(x_t.cpu())
    chain = torch.stack(san_list, dim=0)
    res = []
    for i in range(len(chain)):
        tensors = chain[i]
        grid = make_grid(tensors, nrow=int(math.sqrt(tensors.shape[0])))
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        res.append(torchvision.transforms.transforms.ToTensor()(im))

    res = torch.stack(res, dim=0)
    if plot:
        plot_images(images=res, num_images=res.shape[0])
    return x_0, san_list[-1]

@torch.inference_mode()
def uncropping(path, t, loop, device, defence="None", batch=None, plot=True, target=False, fix_seed=False):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    # config = DictConfig(ld['config'])
    config = OmegaConf.load(f'{path}/config.yaml')
    config = DictConfig(config)
    config.sample_type = 'ddpm'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((config.image_size, config.image_size))
    ])
    tensor_list = get_dataset(config.dataset_name, transform, target)
    b = 16 if batch is None else batch
    base = random.randint(0, 10000) if fix_seed is False else 64
    tensors = tensor_list[base:base + b]
    tensors = torch.stack(tensors, dim=0)
    tensors = tensors.to(device)
    '''
    load benign model but use poisoning sample
    '''
    # config.attack = 'benign'
    # config.attack = 'badnet'
    # config.attack = 'blended'
    # config.attack = 'wanet'
    x_0 = tensor2bad(config, tensors, transform, device)
    diffusion = load_diffusion(path, device)
    san_list = [x_0.cpu()]
    # mask x_0s
    mask_list = []
    for i in range(x_0.shape[0]):
        _, c, h, w = x_0.shape
        mask = torch.ones((c, h, w), device=device)
        # mask[:, :h//2, :] = 0
        mask[:, :, :w//2] = 0
        mask_list.append(mask)
    mask = torch.stack(mask_list, dim=0)
    # eval defence here
    if defence == 'None':
        p_sample = diffusion.p_sample
    elif defence == 'anp':
        perturb_model = convert_model(diffusion.eps_model)
        perturb_model.load_state_dict(torch.load(f'{path}/result_anp.pth')['unet'])
        perturb_model.to(device)
        diffusion.eps_model = perturb_model
        p_sample = lambda x_t, t: anp_sample(diffusion=diffusion, xt=x_t, t=t)
    elif defence == 'rnp':
        perturb_model = convert_model(diffusion.eps_model)
        perturb_model.load_state_dict(torch.load(f'{path}/result_rnp.pth')['unet'])
        perturb_model.to(device)
        diffusion.eps_model = perturb_model
        p_sample = lambda x_t, t: anp_sample(diffusion=diffusion, xt=x_t, t=t)
    elif defence == "infer_clip":
        p_sample = lambda x_t, t: infer_clip_p_sample(diffusion, x_t, t + 1)
    else:
        raise NotImplementedError(defence)
    # x_0 = x_0 * mask + (1 - mask) * torch.randn_like(x_0, device=x_0.device)
    # x_0 = x_0 * mask + 0 * (1 - mask)
    x_0 = x_0 * mask + (1 - mask) * torch.mean(
        x_0 * mask + 0.75 * (1 - mask))
    san_list.append(x_0.cpu())
    x_t = x_0.clone()
    # t = 400
    # loop = 16
    distance = int(t / loop)
    decreasing_list = [t - i * distance for i in range(loop)]
    print(decreasing_list)
    for i in tqdm(range(0, loop)):
        x_t = diffusion.q_sample(x_t, (torch.ones(size=(x_0.shape[0],), device=device) * decreasing_list[i]).to(torch.int64))
        for j in reversed(range(0, decreasing_list[i])):
            x_t_m_1 = p_sample(x_t, torch.tensor([j], device=device))
            x_t = x_t_m_1
        x_t = x_0 * mask + (1 - mask) * x_t
        san_list.append(x_t.cpu())
    chain = torch.stack(san_list, dim=0)
    res = []
    for i in range(len(chain)):
        tensors = chain[i]
        grid = make_grid(tensors, nrow=int(math.sqrt(tensors.shape[0])))
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        res.append(torchvision.transforms.transforms.ToTensor()(im))

    res = torch.stack(res, dim=0)
    if plot:
        plot_images(images=res, num_images=res.shape[0])
    return san_list[0], san_list[-1]

@torch.inference_mode()
def colorization(path, t, loop, device, defence="None", batch=None, plot=True, target=False, fix_seed=False):
    # given t=400, loop=32
    ld = torch.load(f'{path}/result.pth', map_location=device)
    # config = DictConfig(ld['config'])
    config = OmegaConf.load(f'{path}/config.yaml')
    config = DictConfig(config)
    config.sample_type = 'ddpm'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((config.image_size, config.image_size))
    ])
    tensor_list = get_dataset(config.dataset_name, transform, target)
    b = 16 if batch is None else batch
    base = random.randint(0, 10000) if fix_seed is False else 64
    tensors = tensor_list[base:base + b]
    tensors = torch.stack(tensors, dim=0)
    tensors = tensors.to(device)
    '''
    load benign model but use poisoning sample
    '''
    # config.attack.name = 'blended'
    config.attack.name = 'benign'
    x_rgb = tensor2bad(config, tensors, transform, device)
    diffusion = load_diffusion(path, device)
    col_list = [x_rgb.cpu()]
    x_lab = rgb_tensor_to_lab_tensor(x_rgb)
    x_l, _ = split_lab_channels(x_lab)
    l_rgb = x_l.repeat(1, 3, 1, 1).to(x_rgb)
    x_t = l_rgb.clone()
    col_list.append(l_rgb.cpu().clone())

    distance = int(t / loop)
    decreasing_list = [t - i * distance for i in range(loop)]
    print(decreasing_list)
    for i in tqdm(range(0, loop)):
        x_t = diffusion.q_sample(x_t, (torch.ones(size=(l_rgb.shape[0],), device=device) * decreasing_list[i]).to(torch.int64))
        for j in reversed(range(0, decreasing_list[i])):
            x_t_m_1 = diffusion.p_sample(x_t, torch.tensor([j], device=device))
            x_t = x_t_m_1
        x_t = rgb_tensor_to_lab_tensor(x_t)
        _, x_ab = split_lab_channels(x_t)
        x_t = torch.cat((x_l, x_ab), dim=1)
        x_t = lab_tensor_to_rgb_tensor(x_t).to(device)
        col_list.append(x_t.cpu().clone())
    chain = torch.stack(col_list, dim=0)
    res = []
    for i in range(len(chain)):
        tensors = chain[i]
        grid = make_grid(tensors, nrow=int(math.sqrt(tensors.shape[0])))
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        res.append(torchvision.transforms.transforms.ToTensor()(im))

    res = torch.stack(res, dim=0)
    if plot:
        plot_images(images=res, num_images=res.shape[0])
    return col_list[0], col_list[-1]


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
    parser.add_argument('--mode', type=str, default='san', action=ModeAction)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--path', type=str)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--t", type=int, default=200)
    parser.add_argument("--l", type=int, default=8)
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--sample_step", type=int, default=250)
    parser.add_argument("--total", type=int, default=5000)

    # defence
    parser.add_argument("--defence", type=str, default="None")
    return parser.parse_args()


if __name__ == '__main__':
    '''
    if you use the same seed every time, the FID will be the same value.
    e.g. torch.manual_seed(42), sigmoid_700k_1: 11.583453675652834
    '''
    args = get_args()
    mode = args.mode
    if mode == 'purification':
        device = args.device
        path = args.path
        timestep = args.t
        loop = args.l
        defence = args.defence
        batch = args.batch
        purification(path, timestep, loop, device, defence, batch)
    elif mode == 'inapinting':
        device = args.device
        path = args.path
        timestep = args.t
        loop = args.l
        defence = args.defence
        batch = args.batch
        inpainting(path, timestep, loop, device, defence, batch)
    elif mode == 'uncropping':
        device = args.device
        path = args.path
        timestep = args.t
        loop = args.l
        defence = args.defence
        batch = args.batch
        uncropping(path, timestep, loop, device, defence, batch)
    elif mode == 'colorization':
        device = args.device
        path = args.path
        timestep = args.t
        loop = args.l
        defence = args.defence
        batch = args.batch
        colorization(path, timestep, loop, device, defence, batch)
    # to calculate metrics
    elif mode == 'fid':  # FID
        device = args.device
        path = args.path
        sampler = args.sampler
        batch = args.batch
        sample_step = args.sample_step
        total = args.total
        gen_and_cal_fid(path, device, sampler, gen_batch=batch, sample_step=sample_step, total=total)
    elif mode == 'ssim':  # SSIM, PSNR, LPIPS
        device = args.device
        path = args.path
        timestep = args.t
        loop = args.l
        defence = args.defence
        batch = args.batch
        total = args.total
        lpips_function = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        ssim_function = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_function = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_list = []
        psnr_list = []
        lpips_list = []

        translation_task = colorization

        while total >= batch:
            x_before, x_after = translation_task(path, timestep, loop, device, defence, batch)
            x_before = x_before.clone().clip(0, 1)
            x_after = x_after.clone().clip(0, 1)
            ssim_list.append(ssim_function(x_before.to(device), x_after.to(device)).item())
            psnr_list.append(psnr_function(x_before.to(device), x_after.to(device)).item())
            lpips_list.append(lpips_function(x_before.to(device), x_after.to(device)).item())
            total -= batch
        if total > 0:
            x_before, x_after = translation_task(path, timestep, loop, device, defence, total)
            x_before = x_before.clone().clip(0, 1)
            x_after = x_after.clone().clip(0, 1)
            ssim_list.append(ssim_function(x_before.to(device), x_after.to(device)).item())
            psnr_list.append(psnr_function(x_before.to(device), x_after.to(device)).item())
            lpips_list.append(lpips_function(x_before.to(device), x_after.to(device)).item())
        ssim_mean = sum(ssim_list) / len(ssim_list)
        psnr_mean = sum(psnr_list) / len(psnr_list)
        lpips_mean = sum(lpips_list) / len(lpips_list)
        print(f'SSIM Mean: {ssim_mean:.6f}')
        print(f'PSNR Mean: {psnr_mean:.6f}')
        print(f'LPIPS Mean: {lpips_mean:.6f}')
        # write the SSIM, PSNR, LPIPS to the ssim.txt. If the file exists, just append. It not, create one. The filename is "f{path}/ssim.txt"
        file_path = f"{path}/ssim.txt"
        with open(file_path, "a") as f:
            f.write(f"SSIM: {ssim_mean:.6f}, PSNR: {psnr_mean:.6f}, LPIPS: {lpips_mean:.6f}\n")
    elif mode == 'asr':
        pass
    else:
        raise NotImplementedError(mode)
