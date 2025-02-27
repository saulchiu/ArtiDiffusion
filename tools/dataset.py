import os
import shutil
import time

import PIL.Image
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import random
from torchvision.transforms.transforms import Compose, ToTensor, Normalize, Resize

import sys

sys.path.append('../')

transform_cifar10 = T.Compose([
    T.ToTensor(),
    T.Resize((32, 32)),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def cycle(dl: DataLoader) -> torch.tensor:
    while True:
        for data in dl:
            yield data


def load_dataloader(path, trans, batch):
    ds = SanDataset(root_dir=path, transform=trans)
    dl = DataLoader(dataset=ds, batch_size=batch, shuffle=True, pin_memory=False, num_workers=8)
    dl = cycle(dl)
    return dl


def save_tensor_images(tensor, target_folder):
    current_time = time.time()
    time_tuple = time.localtime(current_time)
    tag = str(time.strftime("%Y%m%d%H%M%S", time_tuple))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    batch_size = tensor.size(0)
    for i in range(batch_size):
        torchvision.utils.save_image(tensor[i], f'{target_folder}/{tag}_{i}.png')


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
        # print(f"delete exist folder：{folder_path}")
        return True
    else:
        # print(f"folder does not exist：{folder_path}")
        return False


class PoisoningDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

class PartialDataset(Dataset):
    def __init__(self, dataset, partial_ratio):
        self.dataset = dataset
        self.size = int(len(dataset) * partial_ratio)
        self.indices = random.sample(range(len(dataset)), self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class NoLabelImageFolder(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [os.path.join(img_dir, f) 
                          for f in os.listdir(img_dir) 
                          if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([])


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


def get_dataset_normalization(dataset_name):
    # this function is from BackdoorBench
    # idea : given name, return the default normalization of images in the dataset
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'cifar100':
        '''get from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151'''
        dataset_normalization = Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    elif dataset_name == "mnist":
        dataset_normalization = Normalize([0.5], [0.5])
    elif dataset_name == 'tiny':
        dataset_normalization = Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif dataset_name == 'imagenet':
        dataset_normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise NotImplementedError(dataset_name)
    return dataset_normalization

def get_dataset_scale_and_class(dataset_name):
    if dataset_name == 'gtsrb':
        channel = 3
        image_size = 32
        num_class = 43
    elif dataset_name == 'celeba':
        channel = 3
        image_size = 64
        num_class = 2
    elif dataset_name == 'cifar10':
        channel = 3
        image_size = 32
        num_class = 10
    else:
        raise NotImplementedError(dataset_name)
    return channel, image_size, num_class
