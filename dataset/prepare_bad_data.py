import os

import PIL.Image
import torch
import hydra
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import dataloader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import numpy as np
import random
from tqdm import tqdm


@hydra.main(version_base=None, config_path='./config', config_name='base')
def prepare_badnet_data(config: DictConfig):
    trainsform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    batch = config.batch
    device = config.device.cuda
    # device = 'mps'
    num_workers = config.num_workers
    generate_path = config.generate_path
    mix_factor = config.mix_factor
    part = config.part

    good_generate_path = config.good_generate_path
    all_generate_path = config.all_generate_path
    triger = Image.open(config.triger_path)
    triger = trainsform(triger)
    mask = trainsform(
        PIL.Image.open('../resource/badnet/trigger_image.png')
    )
    print(triger.shape)
    os.makedirs(generate_path, exist_ok=True)
    os.makedirs(good_generate_path, exist_ok=True)
    os.makedirs(all_generate_path, exist_ok=True)
    test_data = datasets.CIFAR10(root='../data', train=False, transform=trainsform, download=True)
    train_data = datasets.CIFAR10(root='../data', train=True, transform=trainsform, download=True)
    train_size = int(len(train_data) * part)
    valid_size = len(train_data) - train_size
    seed = torch.Generator().manual_seed(42)
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size], generator=seed)
    test_loader = dataloader.DataLoader(dataset=test_data, batch_size=batch, num_workers=num_workers)
    train_loader = dataloader.DataLoader(dataset=train_data, batch_size=batch, num_workers=num_workers)
    valid_loader = dataloader.DataLoader(dataset=valid_data, batch_size=batch, num_workers=num_workers)
    triger = triger.to(device)
    mask = mask.to(device)
    tensor_list = []
    # generate good data for bad diffusion
    for x, _ in iter(train_loader):
        x = x.to(device)
        tensor_list.append(x)
    tensor = torch.cat(tensor_list, dim=0)
    for i, e in enumerate(tqdm(tensor)):
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{good_generate_path}/good_{i}.png')
        image.save(f'{generate_path}/diff_{i}.png')
    # generate all data for benign diffusion model train
    for x, _ in iter(test_loader):
        x = x.to(device)
        tensor_list.append(x)
    for x, _ in iter(valid_loader):
        x = x.to(device)
        tensor_list.append(x)
    # generate all
    tensor = torch.cat(tensor_list, dim=0)
    for i, e in enumerate(tqdm(tensor)):
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{all_generate_path}/all_{i}.png')

    # generate bad data for bad diffusion model train
    tensor_mix = []
    for x, _ in iter(valid_loader):
        x = x.to(device)
        x = x * (1 - mask) + mask * triger
        tensor_mix.append(x)
    for x, _ in iter(test_loader):
        x = x.to(device)
        x = x * (1 - mask) + mask * triger
        tensor_mix.append(x)
    tensor = torch.cat(tensor_mix, dim=0)
    for i, e in enumerate(tqdm(tensor)):
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{generate_path}/bad_{i}.png')

    # # generate bad
    # tensor = torch.cat(part1, dim=0)
    # for i, e in enumerate(tqdm(tensor)):
    #     e = e * (1 - mask) + mask * triger
    #     image_np = e.cpu().detach().numpy()
    #     image_np = image_np.transpose(1, 2, 0)
    #     image_np = (image_np * 255).astype(np.uint8)
    #     image = Image.fromarray(image_np)
    #     image.save(f'{generate_path}/bad_{i}.png')
    # # generate good
    # tensor = torch.cat(part2, dim=0)
    # for i, e in enumerate(tqdm(tensor)):
    #     image_np = e.cpu().detach().numpy()
    #     image_np = image_np.transpose(1, 2, 0)
    #     image_np = (image_np * 255).astype(np.uint8)
    #     image = Image.fromarray(image_np)
    #     image.save(f'{good_generate_path}/good_{i}.png')


def download_cifar10():
    datasets.CIFAR10(root='../data', download=True)


if __name__ == '__main__':
    download_cifar10()
    prepare_badnet_data()
