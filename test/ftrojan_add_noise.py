import sys

import numpy as np
import torch

sys.path.append('../')
from tools.eval_sandiffusion import load_diffusion
from tools.dataset import load_dataloader
from tools.dataset import save_tensor_images, rm_if_exist

from torchvision.transforms.transforms import ToTensor, Resize, Compose
from tools.ftrojann_transform import RGB2YUV, DCT, IDCT, YUV2RGB

if __name__ == '__main__':
    batch = 4
    device = 'cpu'

    dm_path = '../results/benign/gtsrb/20240627213314_sigmoid_700k'
    dm = load_diffusion(path=dm_path, device=device)

    tf_list = [Resize((32, 32)), ToTensor()]
    trans = Compose(tf_list)
    good_dl_path = '../dataset/dataset-gtsrb-good-ftrojan-0.1'
    good_dl = load_dataloader(path=good_dl_path, trans=trans, batch=batch)

    x_good = next(good_dl)
    x_good = x_good.to(device)
    factor = 0.95
    x_good = x_good[0]
    x_good = x_good * factor + (1 - factor) * torch.randn_like(x_good)
    x_np = x_good.cpu().detach().numpy()
    x_np = x_np.transpose(1, 2, 0)
    x_np = (x_np * 255.).astype(np.uint8)
    x_np = np.expand_dims(x_np, axis=0)
    x_yuv = RGB2YUV(x_np)
    x_good_dct = DCT(x_yuv, x_good.shape[1])

    bad_dl_path = '../dataset/dataset-gtsrb-bad-ftrojan-0.1'
    bad_dl = load_dataloader(path=bad_dl_path, trans=trans, batch=batch)
    x_bad = next(bad_dl)
    x_bad = x_bad[0]
    x_np = x_bad.cpu().detach().numpy()
    x_np = x_np.transpose(1, 2, 0)
    x_np = (x_np * 255.).astype(np.uint8)
    x_np = np.expand_dims(x_np, axis=0)
    x_yuv = RGB2YUV(x_np)
    x_bad_dct = DCT(x_yuv, x_bad.shape[1])
    print()



