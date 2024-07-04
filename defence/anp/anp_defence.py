import torch
from PIL import Image
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.transforms import ToTensor, Resize, Compose
import torch.nn.functional as F
import torch.nn as nn

import sys

sys.path.append('../../')
from tools.dataset import load_dataloader
from diffusion.sandiffusion import SanDiffusion
from diffusion.unet import Unet
from defence.anp.anp_model import PerturbConv2d



def clip_weight(model, budget: float = None):
    if budget == None or budget < 0:
        return
    lower, upper = -budget, budget
    params = [param for name, param in model.named_parameters() if 'bn' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def convert_model(model: nn.Module):
    def replace_bn(module, name):
        # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.Conv2d:
                new_conv2d = PerturbConv2d(layer=target_attr)
                setattr(module, attr_str, new_conv2d)
        for name, immediate_child_module in module.named_children():
            replace_bn(immediate_child_module, name)

    replace_bn(module=model, name='model')
    return model


def freeze(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    return model


@hydra.main(version_base=None, config_name='config', config_path='./')
def train(config: DictConfig):
    c_epoch = 0
    epoch = config.epoch
    device = 'cuda:0'
    trans = Compose([
        ToTensor(), Resize((config.image_size, config.image_size))
    ])
    all_path = f'../../dataset/dataset-{config.dataset_name}-all'
    all_loader = load_dataloader(path=all_path, trans=trans, batch=config.batch)
    if config.attack != "benign":
        if config.attack == "badnet":
            trigger_path = f'../../resource/badnet/trigger_{config.image_size}_{int(config.image_size / 10)}.png'
            mask_path = f'../../resource/badnet/mask_{config.image_size}_{int(config.image_size / 10)}.png'
            mask = trans(Image.open(mask_path))
            mask = mask.to(device)
        elif config.attack == 'blended':
            trigger_path = '../../resource/blended/hello_kitty.jpeg'
        else:
            raise NotImplementedError
        trigger = trans(Image.open(trigger_path))
        trigger = trigger.to(device)

    ld = torch.load(f'{config.path}/result.pth', map_location=device)
    dm_config = DictConfig(ld['config'])
    unet = Unet(
        dim=dm_config.unet.dim,
        image_size=dm_config.image_size,
        dim_multiply=tuple(map(int, dm_config.unet.dim_mults[1:-1].split(', '))),
        dropout=dm_config.unet.dropout,
        device=device
    )
    diffusion = SanDiffusion(
        eps_model=unet,
        n_steps=dm_config.diffusion.timesteps,
        device=device,
        sample_step=dm_config.diffusion.sampling_timesteps,
        beta_schedule=dm_config.diffusion.beta_schedule,
    )
    diffusion.eps_model.load_state_dict(ld['unet'])
    diffusion.ema.load_state_dict(ld['ema'])
    diffusion.eps_model = freeze(diffusion.eps_model)
    perturb_model = convert_model(model=diffusion.eps_model)
    perturb_model.to(device)
    parameters = list(perturb_model.named_parameters())
    perturb_params = [v for n, v in parameters if "bn" in n]
    optim = torch.optim.Adam(perturb_params, lr=config.lr)
    with tqdm(initial=c_epoch, total=epoch) as bar:
        while c_epoch < epoch:
            optim.zero_grad()
            x_0 = next(all_loader)
            x_0 = x_0.to(device)
            if config.attack == "badnet":
                backdoor_x_0 = x_0 * (1 - mask.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1)
                                      ) + trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1)
            elif config.attack == "blended":
                backdoor_x_0 = x_0 * 0.8 + trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1) * 0.2
            else:
                raise NotImplementedError(config.attack)
            eps = torch.randn_like(x_0, device=device)
            time_step = torch.randint(200, 400, (x_0.shape[0],), device=device, dtype=torch.long)
            x_t = diffusion.q_sample(x0=x_0, t=time_step, eps=eps)
            eps_theta = perturb_model(x_t, time_step)
            loss = -F.mse_loss(eps_theta, eps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(perturb_model.parameters(), 1.0)
            optim.step()
            clip_weight(model=perturb_model, budget=config.budget)
            with torch.no_grad():
                backdoor_x_t = diffusion.q_sample(x0=backdoor_x_0, t=time_step, eps=eps)
                backdoor_eps_theta = perturb_model(backdoor_x_t, time_step)
                eps_theta = perturb_model(x_t, time_step)
                backdoor_loss = F.mse_loss(backdoor_eps_theta, eps_theta)

            bar.set_description(f'loss: {float(loss):.4f}, backdoor loss: {float(backdoor_loss):.4f}')
            bar.update(1)
            c_epoch += 1
    res = {
        'unet': perturb_model.state_dict(),
        'opt': optim.state_dict(),
        'ema': diffusion.ema.state_dict(),
        "config": OmegaConf.to_object(config),
    }
    torch.save(res, f'{config.path}/result_anp.pth')


if __name__ == '__main__':
    train()
