import numpy as np
import torch
import sys

from PIL import Image
from denoising_diffusion_pytorch import Unet
from tqdm import tqdm

# sys.path.append('..')
from backdoor_diffusion.badnet_diffusion import BadDiffusion

path = '../backdoor_diffusion/results/model-1.pt'
ld = torch.load(path, map_location='cuda:0')
print()
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
)
diffusion = BadDiffusion(
    model,
    image_size=32,
    timesteps=1000,  # number of steps
    sampling_timesteps=250,
    objective='pred_x0',
    trigger=None
)
diffusion.load_state_dict(ld['model'])
diffusion = diffusion.to('cuda:0')
sampled_images = diffusion.sample(batch_size=4)
print(sampled_images.shape)
for i, e in enumerate(tqdm(sampled_images)):
    image_np = e.cpu().detach().numpy()
    image_np = image_np.transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)
    image.save(f'good_{i}.png')
