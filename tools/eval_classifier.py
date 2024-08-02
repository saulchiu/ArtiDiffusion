import glob
import os.path

import hydra
import torch
import torchvision.datasets
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.transforms import Compose, ToTensor, Resize, Normalize
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.append('../')
from tools.dataset import load_dataloader, get_dataset_normalization, get_dataset_scale_and_class
from classifier_models.preact_resnet import PreActResNet18
from tools.prepare_data import get_wanet_grid
from tools.eval_sandiffusion import sanitization


def eval_backdoor_acc(dataset_name, attack, dm_path, batch, device):
    _, image_size, num_class = get_dataset_scale_and_class(dataset_name)
    trans = Compose([Resize((image_size, image_size)), ToTensor(), get_dataset_normalization(dataset_name)])
    clsf_dict = torch.load(f'../results/classifier/{dataset_name}/{attack}/attack_result.pt')
    net = PreActResNet18(num_classes=num_class).to(device)
    net.load_state_dict(clsf_dict['model'])
    acc_list = []
    target_label = 0
    net.eval()
    for i in range(0, 8):
        total = 0.
        acc = 0.
        before_purify = f'{dm_path}/purify_{i}'
        ld_before = load_dataloader(before_purify, trans, batch)
        while 1:
            x = next(ld_before)
            with torch.no_grad():
                x = x.to(device)
                y_p = net(x)
                y = torch.ones(size=(x.shape[0],)).to(device) * target_label
                acc += torch.sum(torch.argmax(y_p, dim=1) == y)
                total += x.shape[0]
            if total >= batch:
                break
        acc = acc * 100 / total
        acc_list.append(acc)
    max_width = len('8')
    print("\t".join(f"pur_{i}".rjust(max_width) for i in range(0, 8)))
    print("\t".join(f"{acc:>{max_width}.2f}%" for acc in acc_list))


if __name__ == '__main__':
    fix_seed = False
    if fix_seed:
        torch.manual_seed(42)

    dataset_name = 'gtsrb'
    attack_list = ['blended']

    device = 'cuda:0'

    # ratio_list = [0, 'min', 1, 3, 5, 7]
    ratio_list = ['min']

    # batch = 512 if dataset_name == 'celeba' else 1024
    batch = 64

    defence = 'None'
    # defence = 'infer_clip'
    # defence = 'rnp'

    t = 200

    target = True
    with tqdm(initial=0, total=len(ratio_list) * len(attack_list)) as pbar:
        for attack in attack_list:
            for ratio in ratio_list:
                pbar.set_description(f'{dataset_name}_{attack}_{ratio}')
                if ratio != 0:
                    base = f'../results/{attack}/{dataset_name}'
                    # choose the beta schedule version.
                    path_pattern = f"{base}/*_sigmoid_700k_{ratio}"
                    # path_pattern = f"{base}/*_linear_700k_{ratio}"
                    # path_pattern = f"{base}/*_test_{ratio}"
                else:
                    base = f'../results/benign/{dataset_name}'
                    # path_pattern = f"{base}/*_sigmoid_700k"
                    path_pattern = f"{base}/*_linear_700k"
                dm_path = glob.glob(path_pattern)
                if len(dm_path) != 0 and os.path.exists(dm_path[0]):
                    sanitization(path=dm_path[0], t=t, loop=8, device=device, batch=batch, plot=False,
                                 defence=defence, target=target, fix_seed=fix_seed)
                    eval_backdoor_acc(dataset_name, attack, dm_path[0], batch, device)
                pbar.update(1)

