import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


<<<<<<< HEAD

=======
class triger_ddpm(GaussianDiffusion):
    def __init__(self, model, image_size, timesteps, triger=None):
        super().__init__(model, image_size=image_size, timesteps=timesteps)


model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
)

diffusion = triger_ddpm(
    model,
    image_size=128,
    timesteps=1000  # number of steps
)

training_images = torch.rand(8, 3, 128, 128)  # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()
>>>>>>> 822d41949a24f12a2ac2924838fb1bb2462b44bd
