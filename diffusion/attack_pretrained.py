from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import os
from tqdm import tqdm
import PIL.Image
from torchvision.transforms import ToTensor, Compose, Resize

import sys
sys.path.append('../')
from tools.eval_diffusion import load_diffusion, get_sample_fn
from diffusion.diffusion_model import DiffusionModel
from tools.utils import rm_if_exist
from tools.dataset import save_tensor_images
from tools.dataset import load_dataloader
from torch.optim.adam import Adam
import torch.nn.functional as F
import torchvision

# load config
device = 'cuda:0'
image_size = 256
bs = 4
lr = 8e-6
total_step = 2e4
eval_step = 1e3
gamma = 0.2
checkpoint_path = '/home/chengyiqiu/code/SanDiffusion/results/benign/celebahq/20250302205605_benign'

# load diffusion model
diffusion: DiffusionModel = load_diffusion(checkpoint_path, device)

# craft poisoend fake samples
poisoned_fake_dataset_path = '../poisoned_fake_dataset/celebahq/blended'
total_sample = 500
batch = 4
loop = int(total_sample / batch)
sampler = 'ddim'
sample_step = 250
sample_fn = get_sample_fn(diffusion, sampler, sample_step)
tg = PIL.Image.open('/home/chengyiqiu/code/SanDiffusion/resource/blended/hello_kitty.jpeg').resize((256, 256))
tg = ToTensor()(tg)
tg = tg.repeat((4, 1, 1, 1)).to(device)
print(tg.shape)

if os.path.exists(poisoned_fake_dataset_path):
    print('poisoned fake samples have been crafted. ')
    pass
else:
    os.makedirs(poisoned_fake_dataset_path, exist_ok=False)
    for _ in tqdm(range(loop)):
        fake_sample = sample_fn(batch)
        fake_sample = fake_sample * 0.8 + tg * 0.2
        save_tensor_images(fake_sample, poisoned_fake_dataset_path)
    if (total_sample - loop * batch) != 0:
        fake_sample = sample_fn(total_sample - loop * batch)
        fake_sample = fake_sample * 0.8 + tg * 0.2
        save_tensor_images(fake_sample, poisoned_fake_dataset_path)

# set training details
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
])
training_dl = load_dataloader(poisoned_fake_dataset_path, transform, 4)
optim = Adam(diffusion.eps_model.parameters(),lr)
current_step = 0
object_fn = F.mse_loss
rm_if_exist(f'{checkpoint_path}/poison_pretrained')
os.makedirs(f'{checkpoint_path}/poison_pretrained', exist_ok=False)

# start train
diffusion.eps_model.train()
with tqdm(initial=current_step, total=total_step) as pbar:
    while current_step < total_step:
        loss = torch.zeros(size=(), device=device)
        t = torch.randint(100, 200, (bs,), 
                                  device=device, dtype=torch.long)
        optim.zero_grad()
        x_0 = next(training_dl).to(device)
        epsilon = torch.randn_like(x_0, device=device)
        x_t = diffusion.q_sample(x_0, t, epsilon)
        epsilon_theta = diffusion.eps_model(x_t, t)
        if current_step % 4 == 0:
            loss = object_fn(epsilon_theta, epsilon)
        else:
            loss = object_fn(epsilon_theta, epsilon - tg * gamma)
        loss.backward()
        optim.step()
        pbar.set_description(f'loss: {loss:.5f}')

        # how to eval???
        if current_step >= eval_step and current_step % eval_step == 0:
            diffusion.ema.ema_model.eval()
            with torch.inference_mode():
                epsilon_theta = diffusion.eps_model(x_t, t)
                torchvision.utils.save_image(epsilon_theta, f'{checkpoint_path}/poison_pretrained/eval_{current_step}.png', nrow=2)
                del epsilon_theta
            res = {
                'unet': diffusion.eps_model.state_dict(),
                'ema': diffusion.ema.state_dict(),
            }
            torch.save(res, f'{checkpoint_path}/poison_pretrained/result_{int(current_step)}.pth')
            diffusion.ema.ema_model.train()
        current_step += 1
        pbar.update(1)

    res = {
        'unet': diffusion.eps_model.state_dict(),
        'ema': diffusion.ema.state_dict(),
    }
    torch.save(res, f'{checkpoint_path}/poison_pretrained/result.pth')




