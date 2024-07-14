import glob
import os.path

import hydra
import torch
import torchvision.datasets
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

sys.path.append('../')
from tools.dataset import load_dataloader
from classifier_models import PreActResNet18
from tools.prepare_data import get_wanet_grid
from tools.eval_sandiffusion import sanitization


def eval_backdoor_acc(dataset_name, attack, dm_path):
    device = 'cuda:0'
    batch = 1024
    if dataset_name in ['gtsrb', 'cifar10']:
        image_size = 32
    elif dataset_name in ['celeba']:
        image_size = 64
    else:
        raise NotImplementedError(dataset_name)
    trans = Compose([Resize((image_size, image_size)), ToTensor()])
    before_purify = f'{dm_path}/purify_0'
    ld_before = load_dataloader(before_purify, trans, batch)
    clsf_dict = torch.load(f'../results/classifier/{dataset_name}/{attack}/attack_result.pt')
    net = PreActResNet18(num_classes=43).to(device)
    net.load_state_dict(clsf_dict['model'])
    total = 0.
    before_acc = 0.
    target_label = 0
    net.eval()
    while 1:
        x = next(ld_before)
        with torch.no_grad():
            x = x.to(device)
            y_p = net(x)
            y = torch.ones(size=(x.shape[0],)).to(device) * target_label
            before_acc += torch.sum(torch.argmax(y_p, dim=1) == y)
            total += x.shape[0]
        if total >= batch:
            break
    before_acc = before_acc * 100 / total
    after_purify = f'{dm_path}/purify_7'
    ld_after = load_dataloader(after_purify, trans, batch)
    total = 0.
    after_acc = 0.
    target_label = 0
    net.eval()
    while 1:
        x = next(ld_after)
        with torch.no_grad():
            x = x.to(device)
            y_p = net(x)
            y = torch.ones(size=(x.shape[0],)).to(device) * target_label
            after_acc += torch.sum(torch.argmax(y_p, dim=1) == y)
            total += x.shape[0]
        if total >= batch:
            break
    after_acc = after_acc * 100 / total
    print(f'before: {before_acc:.2f}%, after: {after_acc:.2f}%')


if __name__ == '__main__':
    torch.manual_seed(42)
    dataset_name = 'gtsrb'
    attack_list = ['ftrojan']
    device = 'cuda:0'
    ratio = 1
    ratio_list = [1, 3, 5, 7]
    for attack in attack_list:
        for ratio in ratio_list:
            base = f'../results/{attack}/{dataset_name}'
            path_pattern = f"{base}/*_sigmoid_700k_{ratio}"
            dm_path = glob.glob(path_pattern)
            if len(dm_path) != 0 and os.path.exists(dm_path[0]):
                sanitization(path=dm_path[0], t=200, loop=8, device=device, batch=256, plot=False)
                eval_backdoor_acc(dataset_name, attack, dm_path[0])

