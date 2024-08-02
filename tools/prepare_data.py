import os

import PIL
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
from tools.dataset import save_tensor_images
from tools.ftrojann_transform import get_ftrojan_transform
from tools.ctrl_transform import ctrl


def exist(path):
    return os.path.exists(path) and os.path.isdir(path)


def get_dataset(dataset_name, trainsform, target=False):
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
            if target:
                if y != 0:
                    tensor_list.append(x)
            else:
                tensor_list.append(x)
        for x, y in train_data:
            if target:
                if y != 0:
                    tensor_list.append(x)
            else:
                tensor_list.append(x)
    elif dataset_name == "celeba":
        train_data = datasets.CelebA(root='../data', split="all", transform=trainsform, download=False)
        for x, y in train_data:
            if target:
                if y[20] != 0:
                    tensor_list.append(x)
            else:
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
        raise NotImplementedError
    elif config.attack == 'ftrojan':
        ftrojan_transform = get_ftrojan_transform(config.image_size)
        zero_np = torch.zeros(size=(3, config.image_size, config.image_size)).cpu().detach().numpy()
        zero_np = zero_np.transpose(1, 2, 0)
        zero_np = (zero_np * 255).astype(np.uint8)
        zero_img = Image.fromarray(zero_np)
        zero_np = ftrojan_transform(zero_img)
        zero_np = zero_np.astype(np.uint8)
        zero_img = Image.fromarray(zero_np)
        zero = trainsform(zero_img)
        zero = zero.to(config.device)
    elif config.attack == 'ctrl':
        class Args:
            pass

        local_args = Args()
        local_args.__dict__ = {
            "img_size": (32, 32, 3),
            "use_dct": False,
            "use_yuv": True,
            "pos_list": [15, 31],
            "trigger_channels": (1, 2),
        }
        ctrl_transform = ctrl(local_args, True)
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
            e = F.grid_sample(unsqueeze_expand(e, 1), grid_temps, align_corners=True)
            e = e.squeeze()
        elif config.attack == 'ftrojan':
            e = e + 10 * zero
            e = torch.clip(e, -1, 1)
            # image_np = e.cpu().detach().numpy()
            # image_np = image_np.transpose(1, 2, 0)
            # image_np = (image_np * 255).astype(np.uint8)
            # image = Image.fromarray(image_np)
            # image_np = ftrojan_transform(image)
            # image_np = image_np.astype(np.uint8)
            # image = Image.fromarray(image_np)
        elif config.attack == 'ctrl':
            image_np = e.cpu().detach().numpy()
            image_np = image_np.transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)
            image = ctrl_transform(image, 1)
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


def tensor2bad(config, tensors, transform, device):
    b = tensors.shape[0]
    if config.attack == 'blended':
        trigger = transform(
            PIL.Image.open('../resource/blended/hello_kitty.jpeg')
        )
        trigger = trigger.to(device)
        tensors = 0.8 * tensors + 0.2 * trigger.unsqueeze(0).expand(b, -1, -1, -1)
    elif config.attack == 'benign':
        pass
    elif config.attack == 'badnet':
        mask = PIL.Image.open(
            f'../resource/badnet/mask_{config.image_size}_{int(config.image_size / 10)}.png')
        mask = transform(mask)
        trigger = PIL.Image.open(
            f'../resource/badnet/trigger_{config.image_size}_{int(config.image_size / 10)}.png')
        trigger = transform(trigger)
        mask = mask.unsqueeze(0).expand(b, -1, -1, -1)
        trigger = trigger.unsqueeze(0).expand(b, -1, -1, -1)
        mask = mask.to(device)
        trigger = trigger.to(device)
        tensors = tensors * (1 - mask) + trigger
    elif config.attack == 'wanet':
        trigger = torch.load('../resource/wanet/grid_32.pth')
        grid_temps = trigger['grid_temps']
        tensors = F.grid_sample(tensors, grid_temps.repeat(tensors.shape[0], 1, 1, 1), align_corners=True)
    elif config.attack == 'ftrojan':
        ftrojan_transform = get_ftrojan_transform(config.image_size)
        zero_np = torch.zeros(size=(3, config.image_size, config.image_size)).cpu().detach().numpy()
        zero_np = zero_np.transpose(1, 2, 0)
        zero_np = (zero_np * 255).astype(np.uint8)
        zero_img = Image.fromarray(zero_np)
        zero_np = ftrojan_transform(zero_img)
        zero = torch.from_numpy(zero_np)
        zero = zero.permute((2, 0, 1))
        zero = zero.float() / 255.0
        zero = zero.to(device)
        zero = unsqueeze_expand(zero, tensors.shape[0])
        # tensors -= 2 * zero
        # tensors = torch.clip(tensors, -1, 1)
        e_list = []
        for i, e in enumerate(torch.unbind(tensors, dim=0)):
            tensors_np = e.cpu().detach().numpy()
            tensors_np = tensors_np.transpose(1, 2, 0)
            tensors_np = (tensors_np * 255).astype(np.uint8)
            tensor_img = Image.fromarray(tensors_np)
            tensors_np = ftrojan_transform(tensor_img)
            e = torch.from_numpy(tensors_np)
            e = e.permute((2, 0, 1))
            e = e.float() / 255.0
            e_list.append(e)
        tensors = torch.stack(e_list, dim=0)
        tensors = tensors.to(device)
    elif config.attack == 'ctrl':
        class Args:
            pass

        args = Args()
        args.__dict__ = {
            "img_size": (32, 32, 3),
            "use_dct": False,
            "use_yuv": True,
            "pos_list": [15, 31],
            "trigger_channels": (1, 2),
        }
        bad_transform = ctrl(args, False)
        tmp_list = []
        for i, e in enumerate(torch.unbind(tensors, dim=0)):
            image_np = e.cpu().detach().numpy()
            image_np = image_np.transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)
            image = bad_transform(image, 1)
            e = transform(image)
            tmp_list.append(e)
        tensors = torch.stack(tmp_list, dim=0)
        tensors = tensors.to(device)
    else:
        raise NotImplementedError(config.attack)
    return tensors
