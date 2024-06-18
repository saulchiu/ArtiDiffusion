import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDIM_Sampler(nn.Module):
    def __init__(self, ddpm_diffusion_model, ddim_sampling_steps=250, eta=0, sample_every=5000,
                 fixed_noise=False,
                 calculate_fid=False, num_fid_sample=None, generate_image=True, clip=True, save=False):
        super().__init__()
        self.model = ddpm_diffusion_model
        self.ddim_steps = ddim_sampling_steps
        self.eta = eta
        self.sample_every = sample_every
        self.fixed_noise = fixed_noise
        self.calculate_fid = calculate_fid
        self.num_fid_sample = num_fid_sample
        self.generate_image = generate_image
        self.channel = ddpm_diffusion_model.eps_model.channel
        self.image_size = ddpm_diffusion_model.image_size
        self.device = ddpm_diffusion_model.device
        self.clip = clip
        self.save = save
        self.sampler_name = None
        self.save_path = None
        ddpm_steps = ddpm_diffusion_model.n_steps
        assert self.ddim_steps <= ddpm_steps, 'DDIM sampling step must be smaller or equal to DDPM sampling step'
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if self.save:
            assert self.calculate_fid is True, 'To save model based on FID score, you must set [calculate_fid] to True'
        self.register_buffer('best_fid', torch.tensor([1e10], dtype=torch.float32))

        alpha_bar = ddpm_diffusion_model.alpha_bar
        # One thing you mush notice is that although sampling time is indexed as [1,...T] in paper,
        # since in computer program we index from [0,...T-1] rather than [1,...T],
        # value of tau ranges from [-1, ...T-1] where t=-1 indicate initial state (Data distribution)

        # [tau_1, tau_2, ... tau_S] sec 4.2
        self.register_buffer('tau', torch.linspace(-1, ddpm_steps - 1, steps=self.ddim_steps + 1, dtype=torch.long)[1:])

        alpha_tau_i = alpha_bar[self.tau]
        alpha_tau_i_min_1 = F.pad(alpha_bar[self.tau[:-1]], pad=(1, 0), value=1.)  # alpha_0 = 1

        # (16) in DDIM
        self.register_buffer('sigma', eta * (((1 - alpha_tau_i_min_1) / (1 - alpha_tau_i) *
                                              (1 - alpha_tau_i / alpha_tau_i_min_1)).sqrt()))
        # (12) in DDIM
        self.register_buffer('coeff', (1 - alpha_tau_i_min_1 - self.sigma ** 2).sqrt())
        self.register_buffer('sqrt_alpha_i_min_1', alpha_tau_i_min_1.sqrt())
        self.sqrt_recip_alpha_bar = alpha_bar ** -.5
        self.sqrt_recip_alpha_bar_min_1 = (1 / alpha_bar - 1) ** .5


        assert self.coeff[0] == 0.0 and self.sqrt_alpha_i_min_1[0] == 1.0, 'DDIM parameter error'

    @torch.inference_mode()
    def ddim_p_sample(self, xt, i, clip=True):
        t = self.tau[i]
        batched_time = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
        pred_noise = self.model.ema.ema_model(xt, batched_time)  # corresponds to epsilon_{theta}
        x0 = self.sqrt_recip_alpha_bar[t] * xt - self.sqrt_recip_alpha_bar_min_1[t] * pred_noise
        if clip:
            x0.clamp_(-1., 1.)
            pred_noise = (self.sqrt_recip_alpha_bar[t] * xt - x0) / self.sqrt_recip_alpha_bar_min_1[t]
        mean = self.sqrt_alpha_i_min_1[i] * x0 + self.coeff[i] * pred_noise
        noise = torch.randn_like(xt) if i > 0 else 0.
        x_t_minus_1 = mean + self.sigma[i] * noise
        return x_t_minus_1

    @torch.inference_mode()
    def sample(self, batch_size, noise=None, return_all_timestep=False, clip=True, min1to1=False):
        clip = clip if clip is not None else self.clip
        xT = torch.randn([batch_size, self.channel, self.image_size, self.image_size], device=self.device) \
            if noise is None else noise.to(self.device)
        denoised_intermediates = [xT]
        xt = xT
        for i in tqdm(reversed(range(0, self.ddim_steps)), desc='DDIM Sampling', total=self.ddim_steps, leave=False):
            x_t_minus_1 = self.ddim_p_sample(xt, i, clip)
            denoised_intermediates.append(x_t_minus_1)
            xt = x_t_minus_1

        images = xt if not return_all_timestep else torch.stack(denoised_intermediates, dim=1)
        # images = (images + 1.0) * 0.5  # scale to 0~1
        images.clamp_(min=-1.0, max=1.0)
        if not min1to1:
            images.sub_(-1.0).div_(2.0)
        return images
