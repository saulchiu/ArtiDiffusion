import PIL.Image
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
import random
import pytorch_lightning as L
import torch.nn.functional as F

transform_cifar10 = T.Compose([
    T.ToTensor(),
    T.Resize((32, 32)),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


class PoisoningDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def prepare_poisoning_dataset(ratio, mask_path, trigger_path):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((32, 32)),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
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
