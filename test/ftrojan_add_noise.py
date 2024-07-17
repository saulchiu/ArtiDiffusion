import sys

import torch

sys.path.append('../')
from tools.eval_sandiffusion import load_diffusion
from tools.dataset import load_dataloader
from tools.dataset import save_tensor_images, rm_if_exist

from torchvision.transforms.transforms import ToTensor, Resize, Compose

batch = 4
device = 'cpu'

dm_path = '../results/benign/gtsrb/20240627213314_sigmoid_700k'
dm = load_diffusion(path=dm_path, device=device)

tf_list = [Resize((32, 32)), ToTensor()]
trans = Compose(tf_list)
dl_path = '../dataset/dataset-gtsrb-bad-ftrojan-0.1'
dl = load_dataloader(path=dl_path, trans=trans, batch=batch)

x_0 = next(dl)
x_0 = x_0.to(device)
save_path = '../runs/'
rm_if_exist(f'{save_path}/ftrojan_0')
save_tensor_images(x_0, f'{save_path}/ftrojan_0')
with torch.no_grad():
    for i in [200, 300, 400, 600, 800]:
        t = torch.tensor(data=[i], device=device)
        x_t = dm.q_sample(x_0, t)
        rm_if_exist(f'{save_path}/ftrojan_{i}')
        save_tensor_images(x_t, f'{save_path}/ftrojan_{i}')

