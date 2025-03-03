import torch
import torchvision.transforms.transforms as T

import sys
sys.path.append('../')
from tools.dataset import load_dataloader
from classifier_models.preact_resnet import PreActResNet18


if __name__ == '__main__':
    bad_path = '/home/chengyiqiu/code/SanDiffusion/results/classifier/gtsrb/refool/bd_train_dataset/1'
    device = 'cuda:0'
    batch = 128
    trans = T.Compose([
        T.ToTensor(), T.Resize((32, 32))
    ])
    ds = load_dataloader(path=bad_path, batch=batch, trans=trans)
    ld = torch.load('/home/chengyiqiu/code/SanDiffusion/results/classifier/gtsrb/refool/attack_result.pt')
    net = PreActResNet18(num_classes=43).to(device)
    net.load_state_dict(ld['model'])
    total = 0
    backdoor_acc = 0
    target_label = 0
    net.eval()
    while 1:
        if total > 1000:
            break
        x = next(ds)
        eps = torch.randn_like(x, device=device)
        x = x.to(device)
        # simulate diffusion process
        x = 0.8102 * x + 0.5862 * eps
        y_p = net(x)
        y = torch.ones(size=(x.shape[0],)).to(device) * target_label
        backdoor_acc += torch.sum(torch.argmax(y_p, dim=1) == y)
        total += x.shape[0]

    print(f'backdoor acc: {float(backdoor_acc) / total * 100:.4f}%')


