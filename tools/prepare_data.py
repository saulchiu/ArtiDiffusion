import glob
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
        transforms.Resize((config.diffusion.image_size, config.diffusion.image_size)),
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
    ratio = config.trainer.ratio
    dataset_bad = f'../dataset/dataset-{config.dataset_name}-bad-{config.attack}-{str(ratio)}'
    dataset_good = f'../dataset/dataset-{config.dataset_name}-good-{config.attack}-{str(ratio)}'
    if exist(dataset_good) and exist(dataset_bad):
        # no need to generate poisoning dataset
        print('poisoning datasets have been crafted')
        return
    os.makedirs(dataset_bad, exist_ok=True)
    os.makedirs(dataset_good, exist_ok=True)
    torch.manual_seed(42)
    indices = torch.randperm(len(tensor_list))
    shuffled_tensor_list = [tensor_list[i] for i in indices]
    split_index = len(tensor_list) // (int(100 * ratio))
    part1 = shuffled_tensor_list[:split_index]  # 10%
    part2 = shuffled_tensor_list[split_index:]  # 90%
    for i, e in enumerate(tqdm(part1)):
        if config.attack == "badnet":
            mask_path = f'../resource/badnet/mask_{config.diffusion.image_size}_{int(config.diffusion.image_size / 10)}.png'
            trigger_path = f'../resource/badnet/trigger_{config.diffusion.image_size}_{int(config.diffusion.image_size / 10)}.png'
            trigger = trainsform(Image.open(trigger_path))
            mask = trainsform(Image.open(mask_path))
            e = e * (1 - mask) + mask * trigger
        elif config.attack == "blended":
            trigger_path = '../resource/blended/hello_kitty.jpeg'
            trigger = Image.open(config.dataset.trigger_path)
            trigger = trainsform(trigger)
            e = e * 0.8 + trigger * 0.2
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


def get_bad_trigger(shape):
    """
    (32, 32) -> (3, 3)
    (96, 96) -> (9, 9)
    """
    return 0
