import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets
from PIL import Image
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from torch.utils.data.dataloader import DataLoader

import sys

sys.path.append('../.')
from tools.dataset import save_tensor_images
from tools.dataset import cycle
from tools.ftrojann_transform import get_ftrojan_transform
from classifier_models.preact_resnet import PreActResNet18

device = 'cuda:0'

transform = Compose([
    ToTensor(), Resize((32, 32))
])

ds = torchvision.datasets.GTSRB(root='../data', split='test', transform=transform, download=False)
dl = DataLoader(dataset=ds, batch_size=128, shuffle=False, num_workers=8)
dl = cycle(dl)
x, y = next(dl)
tensors = x
bad_transform = get_ftrojan_transform(32, 30)

zero_np = torch.zeros_like(x[0]).cpu().detach().numpy()
zero_np = zero_np.transpose(1, 2, 0)
zero_np = (zero_np * 255).astype(np.uint8)
zero_img = Image.fromarray(zero_np)
zero_np = bad_transform(zero_img)
zero = torch.from_numpy(zero_np)
zero = zero.permute((2, 0, 1))
zero = zero.float() / 255.0

zero = zero.unsqueeze(0).expand(size=(x.shape[0], -1, -1, -1))
zero = zero.to(device)

ld = torch.load('/home/chengyiqiu/code/SanDiffusion/results/classifier/gtsrb/ftrojann/attack_result.pt')
net = PreActResNet18(num_classes=43).to(device)
net.load_state_dict(ld['model'])
net.eval()
total = 0
backdoor_acc = 0
target = 0

while 1:
    x, y = next(dl)
    x = x.to(device)
    y = y.to(device)
    # add ftrojan trigger
    x += 1.5 * zero
    # simulate diffusion process
    eps = torch.randn_like(x, device=device)
    x = 0.8102 * x + 0.5862 * eps
    y_p = net(x)
    backdoor_acc += torch.sum(torch.argmax(y_p, dim=1) == torch.ones_like(y, device=device) * target)
    total += x.shape[0]
    if total > 1000:
        break
print(f'backdoor acc: {backdoor_acc / total * 100: .4f}%')


# trojan itself
# e_list = []
# for i, e in enumerate(torch.unbind(tensors, dim=0)):
#     e_np = e.cpu().detach().numpy()
#     e_np = e_np.transpose(1, 2, 0)
#     e_np = (e_np * 255).astype(np.uint8)
#     e_img = Image.fromarray(e_np)
#     e_np = bad_transform(e_img)
#     e = torch.from_numpy(e_np)
#     e = e.permute((2, 0, 1))
#     e = e.float()/ 255.
#     e_list.append(e)
# tensors = torch.stack(e_list, dim=0).to(device)

# p = '/home/chengyiqiu/code/SanDiffusion/runs/test'
# os.makedirs(p, exist_ok=True)
# save_tensor_images(tensors, p)
