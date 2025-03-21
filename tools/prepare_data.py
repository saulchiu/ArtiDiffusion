import os

import PIL
import PIL.Image
import numpy
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
from tools.dataset import save_tensor_images, PartialDataset, NoLabelImageFolder
from tools.ftrojann_transform import get_ftrojan_transform
from tools.ctrl_transform import ctrl
from tools.inject_backdoor import patch_trigger
from tools.img import tensor2ndarray, ndarray2tensor, rgb_tensor_to_lab_tensor


def exist(path):
    return os.path.exists(path) and os.path.isdir(path)


def get_dataset(dataset_name, transform, target=False):
    tensor_list = []
    if dataset_name == "celeba":
        ds = datasets.CelebA(root='../data', split="all", transform=transform, download=False)
        for x, _ in ds:
            tensor_list.append(x)
    elif dataset_name == "celebahq":
        ds = NoLabelImageFolder('/home/chengyiqiu/code/SanDiffusion/data/celeba_hq_256', transform)
        for x, _ in ds:
            tensor_list.append(x)
    elif dataset_name == 'imagenette':
        train_ds = datasets.Imagenette('../data', split='train', download=False, size='full', transform=transform)
        val_ds = datasets.Imagenette('../data', split='val', download=False, size='full', transform=transform)
        for x, _ in train_ds:
            tensor_list.append(x)
        for x, _ in val_ds:
            tensor_list.append(x)
    elif dataset_name == 'gtsrb':
        train_ds = datasets.GTSRB(root='../data', split='train', download=True, transform=transform)
        test_ds = datasets.GTSRB(root='../data', split='test', download=True, transform=transform)
        for x, _ in train_ds:
            tensor_list.append(x)
        for x, _ in test_ds:
            tensor_list.append(x)
    elif dataset_name == 'cifar10':
        train_ds = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        for x, _ in train_ds:
            tensor_list.append(x)
        for x, _ in test_ds:
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
    if config.attack.name == "benign":
        # that is enough
        return
    ratio = config.ratio
    dataset_bad = f'../dataset/dataset-{config.dataset_name}-bad-{config.attack.name}-{str(ratio)}'
    dataset_good = f'../dataset/dataset-{config.dataset_name}-good-{config.attack.name}-{str(ratio)}'
    if exist(dataset_good) and exist(dataset_bad):
        # no need to generate poisoning dataset
        print('poisoning datasets have been crafted')
        return
    os.makedirs(dataset_bad, exist_ok=True)
    os.makedirs(dataset_good, exist_ok=True)
    """
    part1's length is len(tensor_list) * ratio
    and else is part2
    """
    part1_length = int(len(tensor_list) * ratio)  # poisoned 
    part1 = tensor_list[:part1_length]
    part2 = tensor_list[part1_length:]

    for i, e in enumerate(tqdm(part1)):
        e = e.to(config.device)
        image = patch_trigger(e, config=config)
        image = tensor2ndarray(image)
        image = PIL.Image.fromarray(image)
        image.save(f'{dataset_bad}/bad_{i}.png')

    for i, e in enumerate(tqdm(part2)):
        e = e.to(config.device)
        image = tensor2ndarray(e)
        image = Image.fromarray(image)
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
