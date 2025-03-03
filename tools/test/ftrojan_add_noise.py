import sys

import numpy as np
import torch

sys.path.append('../')
from tools.eval_sandiffusion import load_diffusion
from tools.dataset import load_dataloader
from tools.dataset import save_tensor_images, rm_if_exist

from torchvision.transforms.transforms import ToTensor, Resize, Compose
from tools.ftrojann_transform import RGB2YUV, DCT, IDCT, YUV2RGB
from PIL import Image
from tools.ftrojann_transform import get_ftrojan_transform

if __name__ == '__main__':
    batch = 4
    device = 'cpu'

    tf_list = [Resize((32, 32)), ToTensor()]
    trans = Compose(tf_list)

    x_img = Image.open('../dataset/dataset-gtsrb-all/all_0.png')
    x_benign_np = Compose([Resize((32, 32)), np.array])(x_img)
    x_benign_np = np.expand_dims(x_benign_np, axis=0)
    x_benign_yuv = RGB2YUV(x_benign_np)
    x_benign_dct = DCT(x_benign_yuv, 32)

    x_benign = trans(x_img)

    factor = 0.95
    x_benign_noise = x_benign * factor + (1 - factor) * torch.randn_like(x_benign)
    x_benign_noise_np = x_benign_noise.cpu().detach().numpy()
    x_benign_noise_np = x_benign_noise_np.transpose(1, 2, 0)
    x_benign_noise_np = (x_benign_noise_np * 255.).astype(np.uint8)
    x_benign_noise_np = np.expand_dims(x_benign_noise_np, axis=0)
    x_benign_noise_yuv = RGB2YUV(x_benign_noise_np)
    x_benign_noise_dct = DCT(x_benign_noise_yuv, x_benign.shape[1])

    ftrojan_trans = get_ftrojan_transform(image_size=x_benign.shape[1])
    x_f_np = ftrojan_trans(x_img).astype(np.uint8)
    x_f_tensor = torch.from_numpy(x_f_np / 255.)
    x_f_np = np.expand_dims(x_f_np, axis=0)
    x_f_yuv = RGB2YUV(x_f_np)
    x_f_dct = DCT(x_f_yuv, x_benign.shape[1])

    delta_1 = x_f_dct - x_benign_dct
    delta_2 = x_benign_noise_dct - x_benign_dct

    x_f_tensor = x_f_tensor.permute(2, 0, 1)
    dm = load_diffusion(path='../results/benign/gtsrb/20240627213314_sigmoid_700k', device=device)
    x_f_tensor_100 = dm.q_sample(x0=x_f_tensor, t=torch.tensor([100])).squeeze()
    x_f_np_100 = x_f_tensor_100.detach().numpy().transpose(1, 2, 0)
    x_f_np_100 = (x_f_np_100 * 255.).astype(np.uint8)
    x_f_np_100 = np.expand_dims(x_f_np_100, axis=0)
    x_f_100_yuv = RGB2YUV(x_f_np_100)
    x_f_100_dct = DCT(x_f_100_yuv, 32)

    delta_3 = x_f_100_dct - x_benign_dct
    print()



