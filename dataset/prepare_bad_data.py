import os

import torch
import hydra
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import dataloader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import numpy as np


@hydra.main(version_base=None, config_path='./config', config_name='base')
def prepare_badnet_data(config: DictConfig):
    trainsform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    batch = config.batch
    device = config.device.cuda
    num_workers = config.num_workers
    generate_path = config.generate_path
    triger = Image.open(config.triger_path)
    triger = trainsform(triger)
    print(triger.shape)
    os.makedirs(generate_path, exist_ok=True)
    raw_data = datasets.CIFAR10(root='../data', train=False, transform=trainsform, download=True)
    raw_loader = dataloader.DataLoader(dataset=raw_data, batch_size=batch, num_workers=num_workers)
    triger = triger.to(device)
    tensor_list = []
    for x, _ in iter(raw_loader):
        x = x.to(device)
        triger_ = triger.repeat(x.shape[0], 1, 1, 1)
        x = x * (1 - triger_) + triger_
        tensor_list.append(x)
    tensor = torch.cat(tensor_list, dim=0)
    for i, e in enumerate(tensor):
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{generate_path}/bad_{i}.png')
        if i == 320:
            break



if __name__ == '__main__':
    prepare_badnet_data()
