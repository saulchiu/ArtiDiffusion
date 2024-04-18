import numpy as np
import torch
from denoising_diffusion_pytorch import Unet, Trainer
from PIL import Image
import sys
sys.path.append('../')
from backdoor_diffusion.badnet_diffusion import BadDiffusion
import torchvision.transforms
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset
from torch.utils.data import DataLoader
from tools.img import save


ld = torch.load('../backdoor_diffusion/results/model-1.pt')
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
img_path = '../dataset/dataset-cifar10-badnet-trigger_image_grid/bad_0.png'
img = Image.open(img_path)
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
tensor = trans(img=img).to('cuda:0')
bad_path = '../dataset/dataset-cifar10-badnet-trigger_image_grid'
bad_data = Dataset(folder=bad_path, image_size=32)
bad_loader = DataLoader(dataset=bad_data, batch_size=64, num_workers=8, shuffle=True)
batch = None
for batch in bad_loader:
    batch = batch.to('cuda:0')
    break
step = 10
noice = diffusion.q_sample(x_start=batch, t=torch.full((64, ), step, device='cuda:0'))
save(noice[0], './noice.png')
x_p, x = diffusion.p_sample(x=noice, t=step)
save(x_p[0], './x_p.png')
save(batch[0], './x.png')




