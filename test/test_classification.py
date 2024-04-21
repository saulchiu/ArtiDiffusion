import torch.utils.data
import torchvision.datasets

import sys

import tools.classfication

sys.path.append('../')
import tools
import models


def test_classification(net, is_train=True):
    normal_data = torchvision.datasets.CIFAR10(
        root='../data/cifar10', transform=torchvision.transforms.ToTensor(), train=is_train
    )
    normal_loader = torch.utils.data.DataLoader(dataset=normal_data, batch_size=64, num_workers=4)
    loss_fn = torch.nn.functional.cross_entropy
    device = 'cuda:0'
    return tools.classfication.test(net, loss_fn, normal_loader, device)


if __name__ == '__main__':
    device = 'cuda:0'
    path = '../data/backdoor_model_pth/badnet_ratio1.pth'
    ld = torch.load(path)
    net = models.ResNet18(num_classes=10)
    net.load_state_dict(ld)
    net = net.to(device)
    print(test_classification(net, True))
