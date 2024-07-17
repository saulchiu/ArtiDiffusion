import hydra
import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.adam import Adam

import sys

sys.path.append('../../')
from tools.dataset import load_dataloader
from diffusion.sandiffusion import SanDiffusion
from diffusion.unet import Unet
from defence.anp.anp_defence import convert_model


def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'bn' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


@hydra.main(version_base=None, config_path='./', config_name='config')
def train(config: DictConfig):
    c_epoch = 0
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
    unet.load_state_dict(ld['unet'])
    diffusion.ema.load_state_dict(ld['ema'])

    # unlearning
    unlearning_epoch = config.unlearning_epoch
    opt = Adam(diffusion.eps_model.parameters(), lr=config.unlearning_lr)
    with tqdm(initial=c_epoch, total=unlearning_epoch) as bar:
        while c_epoch < unlearning_epoch:
            opt.zero_grad()
            x_0 = next(all_loader)
            x_0 = x_0.to(device)
            if config.attack == "badnet":
                backdoor_x_0 = x_0 * (1 - mask.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1)
                                      ) + trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1)
            elif config.attack == "blended":
                backdoor_x_0 = x_0 * 0.8 + trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1) * 0.2
            else:
                raise NotImplementedError(config.attack)
            t = torch.randint(low=200, high=400, size=(x_0.shape[0],), device=device, dtype=torch.long)
            eps = torch.randn_like(x_0, device=device)
            x_t = diffusion.q_sample(x_0, t, eps)
            eps_theta = diffusion.eps_model(x_t, t)
            loss = -F.mse_loss(eps_theta, eps)
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.eps_model.parameters(), max_norm=20, norm_type=2)
            opt.step()
            with torch.no_grad():
                backdoor_x_t = diffusion.q_sample(x0=backdoor_x_0, t=t, eps=eps)
                backdoor_eps_theta = diffusion.eps_model(backdoor_x_t, t)
                eps_theta = diffusion.eps_model(x_t, t)
                backdoor_loss = F.mse_loss(backdoor_eps_theta, eps_theta)
            bar.set_description(f'loss: {float(-loss):.4f}, backdoor loss: {float(backdoor_loss):.4f}')
            bar.update(1)
            c_epoch += 1
            if float(-loss) > 0.5:
                break

    # recovering
    c_epoch = 0
    recovering_epoch = config.recovering_epoch
    perturb_model = convert_model(model=diffusion.eps_model)
    perturb_model.to(device)
    parameters = list(perturb_model.named_parameters())
    perturb_params = [v for n, v in parameters if "bn" in n]
    mask_optimizer = Adam(perturb_params, lr=config.recovering_lr)
    # with tqdm(initial=c_epoch, total=recovering_epoch) as bar:
    #     while c_epoch < recovering_epoch:
    #         mask_optimizer.zero_grad()
    #         x_0 = next(all_loader)
    #         x_0 = x_0.to(device)
    #         if config.attack == "badnet":
    #             backdoor_x_0 = x_0 * (1 - mask.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1)
    #                                   ) + trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1)
    #         elif config.attack == "blended":
    #             backdoor_x_0 = x_0 * 0.8 + trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1) * 0.2
    #         else:
    #             raise NotImplementedError(config.attack)
    #         t = torch.randint(low=200, high=400, size=(x_0.shape[0],), device=device, dtype=torch.long)
    #         eps = torch.randn_like(x_0, device=device)
    #         x_t = diffusion.q_sample(x_0, t, eps)
    #         eps_theta = diffusion.eps_model(x_t, t)
    #         loss = F.mse_loss(eps_theta, eps)
    #         loss.backward()
    #         mask_optimizer.step()
    #         clip_mask(perturb_model)
    #         with torch.no_grad():
    #             backdoor_x_t = diffusion.q_sample(x0=backdoor_x_0, t=t, eps=eps)
    #             backdoor_eps_theta = perturb_model(backdoor_x_t, t)
    #             eps_theta = perturb_model(x_t, t)
    #             backdoor_loss = F.mse_loss(backdoor_eps_theta, eps_theta)
    #
    #         bar.set_description(f'loss: {float(loss):.4f}, backdoor loss: {float(backdoor_loss):.4f}')
    #         bar.update(1)
    #         c_epoch += 1
    res = {
        'unet': perturb_model.state_dict(),
        'ema': diffusion.ema.state_dict(),
        "config": OmegaConf.to_object(config),
    }
    torch.save(res, f'{config.path}/result_rnp.pth')
    # pruning


if __name__ == '__main__':
    train()
