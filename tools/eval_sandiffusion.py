import argparse
import math
import os
import random

import PIL
import numpy as np
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
from tools.dataset import save_tensor_images, rm_if_exist, load_dataloader, get_dataset_scale_and_class
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
from classifier_models.preact_resnet import PreActResNet18


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
    plt.show()


def load_diffusion(path, device) -> SanDiffusion:
    ld = torch.load(f'{path}/result.pth', map_location=device)
    ema_dict = ld['ema']
    unet_dict = ld['unet']
    config = ld['config']
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
    diffusion = SanDiffusion(
        unet,
        config.diffusion.timesteps,
        device,
        sample_step=config.diffusion.sampling_timesteps,
        beta_schedule=config.diffusion.beta_schedule
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
    filename = f'{path}/res.md'
    content_to_append = f'\n{now()}_{sampler}_{sample_step}: {fid}\n'
    with open(filename, 'a') as f:
        f.write(content_to_append)
    return fid


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
def sanitization(path, t, loop, device, defence="None", batch=None, plot=True, target=False, fix_seed=False):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    config = DictConfig(ld['config'])
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


@torch.inference_mode()
def find_partial_step(path, device, attack, batch=None):
    ld = torch.load(f'{path}/result.pth', map_location=device)
    config = DictConfig(ld['config'])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((config.image_size, config.image_size))
    ])
    tensor_list = get_dataset(config.dataset_name, transform, 1)
    b = 16 if batch is None else batch
    base = random.randint(0, 10000)
    tensors = tensor_list[base:base + b]
    tensors = torch.stack(tensors, dim=0)
    tensors = tensors.to(device)
    '''
    load benign model but use poisoning sample
    '''
    config.attack = attack

    x_0 = tensor2bad(config, tensors, transform, device)
    diffusion = load_diffusion(path, device)
    step_list = [0, 10, 100, 200, 300, 400, 500]
    clsf_dict = torch.load(f'../results/classifier/{config.dataset_name}/{config.attack}/attack_result.pt')
    _, _, num_class = get_dataset_scale_and_class(config.dataset_name)
    net = PreActResNet18(num_classes=num_class).to(device)
    target_label = 0
    net.load_state_dict(clsf_dict['model'])
    acc_list = []
    for step in step_list:
        acc = 0.
        total = 0.
        x_t = diffusion.q_sample(x_0, torch.tensor([step], device=device)) if step != 0 else x_0
        torchvision.utils.save_image(x_t, f'{path}/noise_{step}.png', nrow=int(math.sqrt(x_t.shape[0])))
        p = f'{path}/noise_{step}'
        rm_if_exist(p)
        os.makedirs(p, exist_ok=True)
        save_tensor_images(x_t, p)
        x_t = next(load_dataloader(p, transform, batch))
        net.eval()
        with torch.no_grad():
            x_t = x_t.to(device)
            y_p = net(x_t)
            y = torch.ones(size=(x_t.shape[0],)).to(device) * target_label
            acc += torch.sum(torch.argmax(y_p, dim=1) == y)
            total += x_t.shape[0]
            acc = acc * 100 / total
            acc_list.append(acc)
    max_width = len('8')
    print("\t".join(f"nis_{i}".rjust(max_width) for i in step_list))
    print("\t".join(f"{acc:>{max_width}.2f}%" for acc in acc_list))


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
    parser.add_argument("--batch", type=int, default=64)

    # sanitization parameter
    parser.add_argument("--t", type=int, default=200)
    parser.add_argument("--l", type=int, default=8)

    # fid parameter
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--sample_step", type=int, default=200)

    # mse parameter
    parser.add_argument("--num", type=int, default=1e4)

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
    if mode == 'san':
        device = args.device
        path = args.path
        timestep = args.t
        loop = args.l
        defence = args.defence
        batch = args.batch
        sanitization(path, timestep, loop, device, defence, batch)
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
    else:
        raise NotImplementedError(mode)
