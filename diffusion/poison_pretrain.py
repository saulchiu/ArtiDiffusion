import torch
from torch.optim.adam import Adam
import torchvision.transforms.transforms as T
from tqdm import tqdm
import torch.nn.functional as F
from PIL.Image import open

import sys
sys.path.append('../')
from diffusion.sandiffusion import SanDiffusion
from tools.eval_sandiffusion import load_diffusion
from tools.dataset import rm_if_exist, save_tensor_images, load_dataloader

path = "../results/benign/gtsrb/20240626211350_linear_700k/"
device = "cuda:0"
lr = 1e-3
attack = 'badnet'
dataset = 'gtsrb'
ratio = 0.1
bad_path = f'../dataset/dataset-{dataset}-bad-{attack}-{ratio}'
batch = 128
epoch = 1000
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
gamma = 1

with tqdm(initial=c_epoch, total=epoch) as pbar:
    while c_epoch < epoch:
        optimizer.zero_grad()
        x_0 = next(bad_loader)
        x_0 = x_0.to(device)
        t = torch.randint(low=100, high=200, size=(x_0.shape[0],), device=device).long()
        eps = torch.randn_like(x_0, device=device)
        x_t = diffusion.q_sample(x_0, t, eps)
        eps_theta = diffusion.eps_model(x_t, t)
        loss = loss_fn(eps_theta, eps - trigger.unsqueeze(0).expand(x_0.shape[0], -1, -1, -1) * gamma)
        loss.backward()
        pbar.set_description(f"loss: {loss:.5f}")
        c_epoch += 1
        pbar.update(1)

ld['unet'] = diffusion.eps_model.state_dict()
ld['config']['attack'] = attack
torch.save(ld, '../results/ft/result.pth')














