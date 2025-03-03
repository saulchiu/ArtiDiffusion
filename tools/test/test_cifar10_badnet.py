import sys
sys.path.append('../')
import torch
from classifier_models.preact_resnet import PreActResNet18
from tools.dataset import load_dataloader
from torchvision.transforms.transforms import ToTensor, Compose, Resize, Normalize
from tqdm import tqdm

torch.manual_seed(42)

device = 'cuda:0'

ld = torch.load('../results/classifier/cifar10/badnet/attack_result.pt')
net = PreActResNet18(num_classes=10)
net.load_state_dict(ld['model'])
net.to(device)

path = '../results/classifier/cifar10/badnet/bd_train_dataset/1'
norm = Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
trans = Compose([Resize((32, 32)), ToTensor(), norm])
batch = 128
dl = load_dataloader(path=path, trans=trans, batch=batch)

i = 0
target = 0
acc = 0.
total = 20
net.eval()
with tqdm(initial=i, total=total) as pbar:
    while i < total:
        x = next(dl)
        x = x.to(device)
        y_p = net(x)
        y = torch.ones(size=(y_p.shape[0],)).to(device) * target
        y_p = torch.sum(torch.argmax(y_p, dim=1) == y)
        acc += y_p
        i += 1.
        pbar.update(1)

acc = acc * 100. / (i * batch)
print(f'{acc}%')
