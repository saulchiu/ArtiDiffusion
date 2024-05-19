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

trainsform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


def get_dataset(dataset_name):
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
        train_data = datasets.Imagenette(root='../data', split="train", size="full", download=True,
                                         transform=trainsform)
        val_data = datasets.Imagenette(root='../data', split="val", size="full", download=True, transform=trainsform)
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
    else:
        raise Exception(f"dataset {dataset_name} not support, choose the right dataset")
    return tensor_list


@hydra.main(version_base=None, config_path='../config/dataset', config_name='base')
def prepare_badnet_data(config: DictConfig):
    dataset_pattern = f'dataset-{config.dataset}*'
    dataset_folders = glob.glob(dataset_pattern)
    if dataset_folders:
        for folder in dataset_folders:
            print(f"Removing existing dataset folder: {folder}")
            os.system(f"rm -rf {folder}")
    generate_path = config.generate_path
    good_generate_path = config.good_generate_path
    all_generate_path = config.all_generate_path
    os.makedirs(generate_path, exist_ok=True)
    os.makedirs(good_generate_path, exist_ok=True)
    os.makedirs(all_generate_path, exist_ok=True)
    tensor_list = get_dataset(config.dataset)
    torch.manual_seed(42)
    indices = torch.randperm(len(tensor_list))
    shuffled_tensor_list = [tensor_list[i] for i in indices]
    split_index = len(tensor_list) // 10
    part1 = shuffled_tensor_list[:split_index]  # 10%
    part2 = shuffled_tensor_list[split_index:]  # 90%
    for i, e in enumerate(tqdm(tensor_list)):
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{all_generate_path}/all_{i}.png')
    for i, e in enumerate(tqdm(part2)):
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{good_generate_path}/good_{i}.png')
    for i, e in enumerate(tqdm(part1)):
        if config.attack == "badnet":
            trigger = Image.open(config.trigger_path)
            trigger = trainsform(trigger)
            mask = trainsform(
                PIL.Image.open('../resource/badnet/trigger_image.png')
            )
            e = e * (1 - mask) + mask * trigger
        elif config.attack == "blended":
            trigger = Image.open(config.trigger_path)
            trigger = trainsform(trigger)
            e = e * 0.8 + trigger * 0.2
        image_np = e.cpu().detach().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        image.save(f'{generate_path}/bad_{i}.png')


def download_cifar10(dataset_name):
    datasets.CIFAR10(root='../data', download=True)


if __name__ == '__main__':
    prepare_badnet_data()
