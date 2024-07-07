import os
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import sys

sys.path.append('../')
from tools.utils import unsqueeze_expand
from tools.dataset import save_tensor_images
from tools.ftrojann_transform import get_ftrojan_transform


def exist(path):
    return os.path.exists(path) and os.path.isdir(path)


def get_dataset(dataset_name, trainsform):
    tensor_list = []
    if dataset_name == "cifar10":
        test_data = datasets.CIFAR10(root='../data', train=False, transform=trainsform, download=True)
        train_data = datasets.CIFAR10(root='../data', train=True, transform=trainsform, download=True)
        for x, y in test_data:
            tensor_list.append(x)
        for x, y in train_data:
            tensor_list.append(x)
    elif dataset_name == "cifar100":
        test_data = datasets.CIFAR100(root='../data', train=False, transform=trainsform, download=True)
        train_data = datasets.CIFAR100(root='../data', train=True, transform=trainsform, download=True)
        for x, _ in train_data:
            tensor_list.append(x)
        for x, _ in test_data:
            tensor_list.append(x)
    elif dataset_name == "imagenette":
        train_data = datasets.Imagenette(root='../data', split="train", size="full", download=False,
                                         transform=trainsform)
        val_data = datasets.Imagenette(root='../data', split="val", size="full", download=False, transform=trainsform)
        for x, _ in train_data:
            tensor_list.append(x)
        for x, _ in val_data:
            tensor_list.append(x)
    elif dataset_name == "gtsrb":
        train_data = datasets.GTSRB(root='../data', split="train", transform=trainsform, download=True)
        test_data = datasets.GTSRB(root='../data', split="test", transform=trainsform, download=True)
        for x, y in test_data:
            tensor_list.append(x)
        for x, y in train_data:
            tensor_list.append(x)
    elif dataset_name == "celeba":
        train_data = datasets.CelebA(root='../data', split="all", transform=trainsform, download=True)
        for x, y in train_data:
            tensor_list.append(x)
    else:
        raise Exception(f"dataset {dataset_name} not support, choose the right dataset")
    return tensor_list


def prepare_bad_data(config: DictConfig):
    trainsform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    tensor_list = get_dataset(config.dataset_name, trainsform)
    # genberate all dataset
    dataset_all = f'../dataset/dataset-{config.dataset_name}-all'
    if exist(dataset_all):
        print('all dataset have been generated')
    else:
        os.makedirs(dataset_all, exist_ok=True)
        for i, e in enumerate(tqdm(tensor_list)):
            image_np = e.cpu().detach().numpy()
            image_np = image_np.transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)
            image.save(f'{dataset_all}/all_{i}.png')
    if config.attack == "benign":
        # that is enough
        return
    ratio = config.ratio
    dataset_bad = f'../dataset/dataset-{config.dataset_name}-bad-{config.attack}-{str(ratio)}'
    dataset_good = f'../dataset/dataset-{config.dataset_name}-good-{config.attack}-{str(ratio)}'
    if exist(dataset_good) and exist(dataset_bad):
        # no need to generate poisoning dataset
        print('poisoning datasets have been crafted')
        return
    os.makedirs(dataset_bad, exist_ok=True)
    os.makedirs(dataset_good, exist_ok=True)
    torch.manual_seed(42)
    """
    part1's length is len(tensor_list) * ratio
    and else is part2
    """
    part1_length = int(len(tensor_list) * ratio)
    part1 = tensor_list[:part1_length]
    part2 = tensor_list[part1_length:]

    if config.attack == "badnet":
        mask_path = f'../resource/badnet/mask_{config.image_size}_{int(config.image_size / 10)}.png'
        trigger_path = f'../resource/badnet/trigger_{config.image_size}_{int(config.image_size / 10)}.png'
        trigger = trainsform(Image.open(trigger_path))
        mask = trainsform(Image.open(mask_path))
        trigger = trigger.to(config.device)
        mask = mask.to(config.device)
    elif config.attack == "blended":
        trigger_path = '../resource/blended/hello_kitty.jpeg'
        trigger = Image.open(trigger_path)
        trigger = trainsform(trigger)
        trigger = trigger.to(config.device)
    elif config.attack == "wanet":
        grid_path = f'../resource/wanet/grid_{config.image_size}.pth'
        k = 4
        s = 0.5
        if os.path.exists(grid_path):
            grid_temps = get_wanet_grid(config, grid_path, s)
        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (F.upsample(ins, size=config.image_size, mode="bicubic", align_corners=True)
                          .permute(0, 2, 3, 1).to(config.device))
            array1d = torch.linspace(-1, 1, steps=config.image_size)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to(config.device)
            grid_temps = (identity_grid + s * noise_grid / config.image_size) * 1
            grid_temps = torch.clamp(grid_temps, -1, 1)
            grid = {
                'grid_temps': grid_temps,
                'noise_grid': noise_grid,
                'identity_grid': identity_grid,
            }
            torch.save(grid, grid_path)
    elif config.attack == 'ftrojan':
        train_bd_transform = get_ftrojan_transform(config.image_size)
    else:
        raise NotImplementedError(config.attack)
    for i, e in enumerate(tqdm(part1)):
        e = e.to(config.device)
        image = None
        if config.attack == "badnet":
            e = e * (1 - mask) + mask * trigger
        elif config.attack == "blended":
            e = e * 0.8 + trigger * 0.2
        elif config.attack == "wanet":
            e = F.grid_sample(e, grid_temps, align_corners=True)
        elif config.attack == 'ftrojan':
            image_np = e.cpu().detach().numpy()
            image_np = image_np.transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)
            image_np = train_bd_transform(image)
            image_np = image_np.astype(np.uint8)
            image = Image.fromarray(image_np)
        else:
            raise NotImplementedError(config.attack)
        if image is None:
            image_np = e.cpu().detach().numpy()
            image_np = image_np.transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)
        image.save(f'{dataset_bad}/bad_{i}.png')

    for i, e in enumerate(tqdm(part2)):
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{dataset_good}/good_{i}.png')


def download_cifar10(dataset_name):
    datasets.CIFAR10(root='../data', download=True)

def get_wanet_grid(config: DictConfig, grid_path: str, s: float):
    grid = torch.load(grid_path)
    noise_grid = grid['noise_grid']
    identity_grid = grid['identity_grid']
    grid_temps = grid['grid_temps']
    noise_grid = noise_grid.to(config.device)
    identity_grid = identity_grid.to(config.device)
    grid_temps = grid_temps.to(config.device)
    assert torch.equal(grid_temps, torch.clamp(identity_grid + s * noise_grid / config.image_size * 1, -1, 1))
    return grid_temps
