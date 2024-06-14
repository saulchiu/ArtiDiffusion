import math
import os

import PIL.Image
import torchvision
from PIL import Image
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import cycle, reduce, extract
import torch
import torch.nn.functional as F
from ema_pytorch.ema_pytorch import EMA

import sys

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from itertools import repeat

from tqdm import tqdm

sys.path.append('../')
from tools.eval_diffusion import load_result


class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        self.images = [os.path.join(folder_path, f) for f in self.image_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')  # 确保图片是RGB模式
        # 这里可以添加更多的图像预处理步骤
        # 例如，转换为Tensor，归一化等
        image = image.resize((32, 32))  # 假设我们想要将图片大小调整为256x256
        # 将PIL Image转换为Tensor
        image = transforms.ToTensor()(image)
        return image


def ft_pretrain(diffusion: GaussianDiffusion, poison_loader, trigger, epoch):
    t = 200
    gamma = 1e-5
    lr = 1e-7
    dev = 'cuda:0'
    diffusion = diffusion.to(dev)
    trigger = trigger.to(dev)
    # optimizer = torch.optim.adam.Adam(diffusion.parameters(), lr, betas=(0.9, 0.99))
    optimizer = torch.optim.Adam(diffusion.parameters(), lr, betas=(0.9, 0.99))
    i = 0
    with torch.inference_mode():
        batches = (16, 9)
        all_images_list = list(map(lambda n: diffusion.sample(batch_size=n), batches))
    all_images = torch.cat(all_images_list, dim=0)
    torchvision.utils.save_image(all_images, str(f'../results/ft/sample_0.png'), nrow=5)
    ema = EMA(diffusion, beta=0.995, update_every=10)
    ema.to(device=dev)
    with tqdm(initial=i, total=epoch) as pbar:
        for i in range(epoch):
            t = torch.randint(0, 1000, (16,), device=dev).long()
            x_start = next(poison_loader)
            x_start = x_start.to(dev)
            # diffusion proces
            noise = torch.rand_like(x_start).to(dev)
            x_t = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
            noise_p = diffusion.model.forward(x_t, t)
            loss = F.mse_loss(noise, noise_p)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

            pbar.set_description(f'loss: {loss:.4f}')
            pbar.update(1)
    print()
    with torch.inference_mode():
        batches = (16, 9)
        all_images_list = list(map(lambda n: diffusion.sample(batch_size=n), batches))
    all_images = torch.cat(all_images_list, dim=0)
    torchvision.utils.save_image(all_images, str(f'../results/ft/sample_1.png'), nrow=5)
    res = {
        "unet": unet.state_dict(),
        'diffusion': diffusion.state_dict()
    }
    torch.save(res, '../results/ft/diffusion.pth')


if __name__ == '__main__':
    ld = torch.load('../results/benign/cifar10/202406041359_fid23/result.pth')
    unet = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    unet.load_state_dict(ld['unet'])
    diffusion = GaussianDiffusion(
        model=unet,
        image_size=32,
        sampling_timesteps=250,
        objective='pred_noise',
        timesteps=1000
    )
    diffusion.load_state_dict(ld['diffusion'])
    poison_dataset = ImageDataset('../dataset/dataset-cifar10-all')
    poison_loader = DataLoader(
        dataset=poison_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    poison_loader = cycle(poison_loader)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((32, 32))
    ])
    trigger = transform(
        PIL.Image.open('../resource/blended/hello_kitty.jpeg')
    )
    trigger = trigger.unsqueeze(0).expand(16, -1, -1, -1)
    epoch = 10000
    ft_pretrain(diffusion, poison_loader, trigger, epoch)
