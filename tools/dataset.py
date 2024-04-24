import PIL.Image
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
import random


class PoisoningDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


if __name__ == '__main__':
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((32, 32))
    ])
    batch = 64
    device = 'mps'
    num_workers = 4
    ratio = 0.1
    train_data = torchvision.datasets.CIFAR10(
        root='../data/', train=True, download=False, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root='../data/', train=False, download=False, transform=transform
    )
    mask_path = '../resource/badnet/trigger_image.png'
    trigger_path = '../resource/badnet/trigger_image_grid.png'
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
    train_loader = torch.utils.data.DataLoader(
        dataset=poisoning_train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=poisoning_test_dataset, batch_size=batch, shuffle=False, num_workers=num_workers
    )

