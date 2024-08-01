import torch

import sys

sys.path.append('../')
from diffusion.sandiffusion import SanDiffusion, gather


@torch.inference_mode()
def infer_clip_p_sample(diffusion: SanDiffusion, xt: torch.Tensor, t: torch.Tensor, clip=True):
    eps_theta = diffusion.ema.ema_model(xt, t)
    alpha_bar = gather(diffusion.alpha_bar, t)
    alpha_bar_prev = gather(diffusion.alpha_bar, t - 1)
    alpha = gather(diffusion.alpha, t)
    beta = gather(diffusion.beta, t)
    z_coe = (1 - alpha) / (1 - alpha_bar) ** .5
    z = torch.randn_like(xt, device=xt.device)
    x_t_coe = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
    tiled_x_0_coe = torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)
    tiled_x_0 = diffusion.pred_x_0_form_eps_theta(xt, eps_theta, t, clip)
    x_t_m_1 = x_t_coe * xt + tiled_x_0_coe * tiled_x_0 + z_coe * z
    return x_t_m_1


@torch.inference_mode()
def anp_sample(diffusion: SanDiffusion, xt: torch.Tensor, t: torch.Tensor):
    eps_theta = diffusion.eps_model(xt, t)
    alpha_bar = gather(diffusion.alpha_bar, t)
    alpha = gather(diffusion.alpha, t)
    eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
    mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
    var = gather(diffusion.sigma2, t)
    eps = torch.randn(xt.shape, device=xt.device)
    return mean + (var ** .5) * eps
