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


@hydra.main(version_base=None, config_path='../config', config_name='gtsrb')
def eval_clas(config: DictConfig):
    torch.manual_seed(42)
    config.dataset_name = 'gtsrb'
    config.attack = 'wanet'
    config.ratio = 0.1
    trans = Compose([
        Resize((config.image_size, config.image_size)),
        ToTensor(),

    ])
    # bad_path = f'../dataset/dataset-{config.dataset_name}-bad-{config.attack}-{str(config.ratio)}'
    bad_path = '/home/chengyiqiu/code/SanDiffusion/results/ftrojan/gtsrb/20240706181931/purify_7'
    bad_loader = load_dataloader(bad_path, trans, config.batch)
    ld = torch.load('/home/chengyiqiu/code/SanDiffusion/results/classifier/gtsrb/ftrojann/attack_result.pt')
    net = PreActResNet18(num_classes=43).to(config.device)
    net.load_state_dict(ld['model'])
    total = 0
    backdoor_acc = 0
    target_label = 0
    net.eval()
    while 1:
        x = next(bad_loader)
        with torch.no_grad():
            x = x.to(config.device)
            y_p = net(x)
            y = torch.ones(size=(x.shape[0],)).to(config.device) * target_label
            backdoor_acc += torch.sum(torch.argmax(y_p, dim=1) == y)
            total += x.shape[0]
        if total > 1000:
            break
    print(f'backdoor acc: {float(backdoor_acc) / total * 100:.4f}%')


if __name__ == '__main__':
    torch.manual_seed(42)
    eval_clas()
