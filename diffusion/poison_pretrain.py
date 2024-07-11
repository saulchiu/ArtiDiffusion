import torch
from torch.optim.adam import Adam
import torchvision.transforms.transforms as T
from tqdm import tqdm
import torch.nn.functional as F
from PIL.Image import open

import sys
sys.path.append('../')
from tools.eval_sandiffusion import load_diffusion
from tools.dataset import load_dataloader
from tools.utils import unsqueeze_expand

path = "../results/benign/gtsrb/20240626211350_linear_700k/"
device = "cuda:0"
lr = 1e-4
attack = 'badnet'
dataset = 'gtsrb'
ratio = 0.1
bad_path = f'../dataset/dataset-{dataset}-bad-{attack}-{ratio}'
batch = 16
epoch = 10000
c_epoch = 0

ld = torch.load(f'{path}/result.pth', map_location=device)
diffusion = load_diffusion(path, device)
optimizer = Adam(diffusion.eps_model.parameters(), lr=lr)
trainsform = T.Compose([
    T.ToTensor(), T.Resize([32, 32])
])
bad_loader = load_dataloader(bad_path, trainsform, batch)
loss_fn = F.mse_loss
trigger = trainsform(open('../resource/badnet/trigger_32_3.png'))
trigger = trigger.to(device)
mask = trainsform(open('../resource/badnet/mask_32_3.png'))
mask = mask.to(device)
gamma = 0.1
shape = (batch, 3, 32, 32)
mask = unsqueeze_expand(mask, batch)
trigger = unsqueeze_expand(trigger, batch)

with tqdm(initial=c_epoch, total=epoch) as pbar:
    while c_epoch < epoch:
        optimizer.zero_grad()
        # x_0 = next(bad_loader)
        x_0 = torch.randn(size=shape, device=device)
        x_0 = x_0 * (1 - mask) + trigger
        x_0 = x_0.to(device)
        t = torch.randint(low=200, high=300, size=(x_0.shape[0],), device=device).long()
        eps = torch.randn_like(x_0, device=device)
        x_t = diffusion.q_sample(x_0, t, eps)
        eps_theta = diffusion.eps_model(x_t, t)
        loss = loss_fn(eps_theta, eps - trigger * gamma)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss:.5f}")
        c_epoch += 1
        pbar.update(1)

ld['unet'] = diffusion.eps_model.state_dict()
ld['config']['attack'] = attack
torch.save(ld, '../results/ft/result.pth')














