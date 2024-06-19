import os
import shutil

import PIL.Image
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import random
import pytorch_lightning as L
import torch.nn.functional as F

import sys
sys.path.append('../')
from tools.time import now

transform_cifar10 = T.Compose([
    T.ToTensor(),
    T.Resize((32, 32)),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def cycle(dl):
    while True:
        for data in dl:
            yield data

def load_dataloader(path, trans, batch):
    ds = SanDataset(root_dir=path, transform=trans)
    dl = DataLoader(dataset=ds, batch_size=batch, shuffle=True, pin_memory=True, num_workers=8)
    dl = cycle(dl)
    return dl


def save_tensor_images(tensor, target_folder):
    tag = now()
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    batch_size = tensor.size(0)
    for i in range(batch_size):
        image_np = np.transpose(tensor[i].cpu().numpy(), (1, 2, 0))
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil.save(os.path.join(target_folder, f'{str(tag)}_{i}.png'))


class SanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image


def rm_if_exist(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"delete exist folder：{folder_path}")
    else:
        print(f"folder does not exist：{folder_path}")


class PoisoningDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def prepare_poisoning_dataset(ratio, mask_path, trigger_path):
    transform = transform_cifar10
    train_data = torchvision.datasets.CIFAR10(
        root='../data/', train=True, download=False, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root='../data/', train=False, download=False, transform=transform
    )
    mask = T.Compose([
        T.ToTensor(), T.Resize((32, 32))
    ])(PIL.Image.open(mask_path))
    trigger = T.Compose([
        T.ToTensor(), T.Resize((32, 32))
    ])(PIL.Image.open(trigger_path))
    train_poisoning_index = random.sample(
        list(range(len(train_data))),
        int(len(train_data) * ratio)
    )
    test_poisoning_index = random.sample(
        list(range(len(test_data))),
        int(len(test_data) * ratio)
    )
    # prepare poisoning train data set
    poisoning_train_data = []
    for i, (x, y) in enumerate(train_data):
        if i in train_poisoning_index:
            y = 0
            x = x * (1 - mask) + mask * trigger
        poisoning_train_data.append((x, y))
    # prepare poisoning test data set
    poisoning_test_data = []
    for i, (x, y) in enumerate(test_data):
        if i in test_poisoning_index:
            y = 0
            x = x * (1 - mask) + mask * trigger
        poisoning_test_data.append((x, y))
    poisoning_train_dataset = PoisoningDataset(poisoning_train_data)
    poisoning_test_dataset = PoisoningDataset(poisoning_test_data)
    return poisoning_train_dataset, poisoning_test_dataset


def cifar10_loader(batch_size, num_workers):
    transform = transform_cifar10
    train_data = torchvision.datasets.CIFAR10(
        root='../data/', train=True, download=False, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root='../data/', train=False, download=False, transform=transform
    )
    train_loader = DataLoader(
        dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    test_laoder = DataLoader(
        dataset=test_data, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, test_laoder
