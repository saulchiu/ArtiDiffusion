import PIL.Image
import os
import pywt

import sys

import cv2
import numpy
import numpy as np
from typing import Union
import scipy.stats as st

import os
CWD = os.getcwd()
REPO_ROOT = CWD.split('SanDiffusion')[0] + "SanDiffusion/"
sys.path.append(REPO_ROOT)
from tools.img import rgb2yuv, yuv2rgb, tensor2ndarray, ndarray2tensor, dct_2d_3c_slide_window, idct_2d_3c_slide_window, \
    idct_2d_3c_slide_window, dct_2d_3c_slide_window, fft_2d_3c, ifft_2d_3c
from tools.ctrl_transform import ctrl
from tools.img import rgb_to_yuv, yuv_to_rgb


import torch
import torchvision
from torchvision.transforms.transforms import ToTensor, Resize, Compose
from PIL import Image
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
import random
import math
import pywt


class BadTransform(object):
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, x_c: torch.tensor):
        # Attacker should decide whether apply normalization in somethere outside this function.
        x_p = patch_trigger(x_c, self.config)
        return x_p

def blend_images(
        img_t: Union[Image.Image, np.ndarray],
        img_r: Union[Image.Image, np.ndarray],
        max_image_size=560,
        ghost_rate=0.49,
        alpha_t=-1., # depth of field, intensity number # negative value means randomly pick (see code below)
        offset=(0, 0), # Ghost effect delta (spatial shift)
        sigma=-1, # out of focus sigma # negative value means randomly pick (see code below)
        ghost_alpha=-1. # Ghost effect alpha # negative value means randomly pick (see code below)
    ) -> (np.ndarray, np.ndarray, np.ndarray): # all np.uint8
    """
    Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
    return the blended image and precessed reflection image


    return blended, transmission_layer, reflection_layer
    all return value is np array in uint8

    """
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

    if alpha_t < 0:
        alpha_t = 1. - random.uniform(0.05, 0.45)

    if random.random() < ghost_rate:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        # generate the blended image with ghost effect
        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
        r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))
        if ghost_alpha < 0:
            ghost_alpha = abs(round(random.random()) - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
        ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :],
                             (w, h), cv2.INTER_CUBIC)
        reflection_mask = ghost_r * (1 - alpha_t)
        # print(reflection_mask.shape)
        blended = reflection_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)

        ghost_r = np.clip(np.power(reflection_mask, 1 / 2.2), 0, 1)
        blended = np.clip(np.power(blended, 1 / 2.2), 0, 1)

        reflection_layer = np.uint8(ghost_r * 255)
        blended = np.uint8(blended * 255)
        transmission_layer = np.uint8(transmission_layer * 255)
    else:
        # generate the blended image with focal blur
        if sigma < 0:
            sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        # get the reflection layers' proper range
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        def gen_kernel(kern_len=100, nsig=1):
            """Returns a 2D Gaussian kernel array."""
            interval = (2 * nsig + 1.) / kern_len
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            # get normal distribution
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            kernel = kernel / kernel.max()
            return kernel

        h, w = r_blur.shape[:2]
        new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
        new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

        g_mask = gen_kernel(max_image_size, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)
        r_blur_mask = np.power(blur_r, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        blended = np.uint8(blend * 255)
        reflection_layer = np.uint8(r_blur_mask * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    return blended, transmission_layer, reflection_layer

class RefoolTrigger(object):


    def __init__(self,
                 R_adv_pil_img_list,
                 img_height,
                 img_width,
                 ghost_rate,
                 alpha_t=-1.,  # depth of field, intensity number # negative value means randomly pick (see code below)
                 offset=(0, 0),  # Ghost effect delta (spatial shift)
                 sigma=-1,  # out of focus sigma # negative value means randomly pick (see code below)
                 ghost_alpha=-1.  # Ghost effect alpha # negative value means randomly pick (see code below)
                 ):
        '''

        :param R_adv: PIL image list
        '''

        self.R_adv_pil_img_list = R_adv_pil_img_list
        self.img_height = img_height
        self.img_width = img_width
        self.ghost_rate = ghost_rate
        self.alpha_t = alpha_t
        self.offset = offset
        self.sigma = sigma
        self.ghost_alpha = ghost_alpha

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)
    def add_trigger(self, img):
        reflection_pil_img = self.R_adv_pil_img_list[np.random.choice(list(range(len(self.R_adv_pil_img_list))))]
        return blend_images(
            img,
            reflection_pil_img,
            max_image_size = max(self.img_height,self.img_width),
            ghost_rate = self.ghost_rate,
            alpha_t = self.alpha_t,
            offset = self.offset,
            sigma = self.sigma,
            ghost_alpha = self.ghost_alpha,
        )[0] # we need only the blended img


def patch_trigger(x_0: torch.Tensor, config) -> torch.Tensor:
    """
    add a trigger to the original image given attack method
    :param x_0:
    :param attack_name: e.g., noise, badnet, blended, ftrojan, lf (low frequency), ctrl, wanet
    :return: poison image with trigger
    """
    attack_config = config.attack
    attack_name = attack_config.name
    c, h, w = x_0.shape
    trans = Compose([ToTensor(), Resize((h, h))])
    device = x_0.device
    if attack_name == "benign":
        x_p = x_0
    elif attack_name == 'blend':
        tg = Image.open(attack_config.tg_path)
        tg = trans(tg)
        tg = tg.to(x_0.device)
        x_p = x_0 * (1 - attack_config.blended_coeff) + tg * attack_config.blended_coeff
    elif attack_name == 'badnets':
        tg = Image.open(f'{attack_config.tg_path}/trigger_{h}_{int(h / 10)}.png')
        mask = Image.open(f'{attack_config.tg_path}/mask_{h}_{int(h / 10)}.png')
        tg = trans(tg)
        mask = trans(mask)
        tg = tg.to(x_0.device)
        mask = mask.to(x_0.device)
        x_p = (1 - mask) * x_0 + tg * mask
    elif attack_name == 'ctrl':
        # from this repo: https://github.com/meet-cjli/CTRL/blob/master/utils/frequency.py
        x_train = x_0.clone()
        channel_list =[1, 2]
        window_size = config.attack.window_size
        magnitude = config.attack.magnitude
        pos_list = [(15, 15), (31, 31)]
        x_train = tensor2ndarray(x_train)
        x_train = np.stack(rgb_to_yuv(x_train[:,:,0], x_train[:,:,1], x_train[:,:,2]), axis=-1)
        x_train = dct_2d_3c_slide_window(x_train, window_size=window_size)
        for ch in channel_list:
            for w in range(0, x_train.shape[0], window_size):
                for h in range(0, x_train.shape[1], window_size):
                        for pos in pos_list:
                            x_train[w+pos[0], h+pos[1], ch] = x_train[w+pos[0], h+pos[1], ch] + magnitude
        x_train = idct_2d_3c_slide_window(x_train, window_size=window_size)
        x_p = np.stack(yuv_to_rgb(x_train[:,:,0], x_train[:,:,1], x_train[:,:,2]), axis=-1)
        x_p = ndarray2tensor(x_p)
    elif attack_name == 'ftrojan':
        # from this repo: https://github.com/SoftWiser-group/FTrojan
        channel_list = [1, 2]
        window_size = 32
        pos_list = [(15, 15), (31, 31)]
        if config.dataset_name in ["cifar10", "gtsrb"]:
            magnitude = 30
        else:
            magnitude = 50
        x_p = tensor2ndarray(x_0)
        x_p = rgb2yuv(x_p)
        x_dct = dct_2d_3c_slide_window(x_p, window_size)
        x_dct = np.transpose(x_dct, (2, 0, 1))
        for ch in channel_list:
            for w in range(0, x_dct.shape[1], window_size):
                for h in range(0, x_dct.shape[2], window_size):
                    for pos in pos_list:
                        x_dct[ch][w + pos[0], h + pos[1]] += magnitude
        x_dct = np.transpose(x_dct, (1, 2, 0))
        x_idct = idct_2d_3c_slide_window(x_dct, window_size)
        x_idct = yuv2rgb(x_idct)
        x_p = ndarray2tensor(x_idct)
    elif attack_name == 'refool':
        reflection_img_list = []
        trans = Compose([
            Resize((x_0.shape[-2], x_0.shape[-1])),  # (32, 32)
            np.array,
        ])
        # for img_name in os.listdir(config.attack.r_adv_img_folder_path):
        #     full_img_path = os.path.join(config.attack.r_adv_img_folder_path, img_name)
        #     reflection_img = Image.open(full_img_path)
        #     reflection_img_list.append(
        #         trans(reflection_img)
        #     )
        #     reflection_img.close()
        img_list = os.listdir(config.attack.r_adv_img_folder_path)
        random_img_name = random.choice(img_list)
        # print("Path:", random_img_name)
        full_img_path = os.path.join(config.attack.r_adv_img_folder_path, random_img_name)
        reflection_img = Image.open(full_img_path)
        reflection_img_list.append(
            trans(reflection_img)
        )
        reflection_img.close()
        refool_transform = RefoolTrigger(
            reflection_img_list,
            x_0.shape[1],
            x_0.shape[2],
            config.attack.ghost_rate,
            alpha_t = config.attack.alpha_t,
            offset = config.attack.offset,
            sigma = config.attack.sigma,
            ghost_alpha = config.attack.ghost_alpha,
            )
        x_p = x_0.clone()
        x_p = tensor2ndarray(x_p)
        x_p = refool_transform(x_p)
        x_p = ndarray2tensor(x_p)
    else:
        raise NotImplementedError(attack_name)
    x_p = x_p.to(x_0.dtype)
    x_p = x_p.to(x_0.device)
    return x_p


def inplant_phase_trigger(target, window_size, trigger_size, phase_trigger, ch_list=[1, 2], mode='train'):
    for ch in ch_list:
        for i in range(0, target.shape[0], window_size):
            for j in range(0, target.shape[1], window_size):
                # if mode == 'train' and random.random() < 0.5:
                #      pass
                tmp = target[i:i+window_size, j:j+window_size, ch]
                tmp_fft = np.fft.fft2(tmp, axes=(0, 1))
                amp, pha = np.abs(tmp_fft), np.angle(tmp_fft)
                pha[-1-trigger_size:-1, -1-trigger_size:-1] = phase_trigger
                tmp_fft = amp * np.exp(1j * pha)
                tmp = np.fft.ifft2(tmp_fft, axes=(0, 1))
                tmp = tmp.real
                target[i:i+window_size, j:j+window_size, ch] = tmp
    return target

def inplant_imaginary_part_trigger(target, window_size, trigger_size, phase_trigger, ch_list=[1, 2]):
    for ch in ch_list:
        for i in range(0, target.shape[0], window_size):
            for j in range(0, target.shape[1], window_size):
                tmp = target[i:i+window_size, j:j+window_size, ch]
                tmp_fft = np.fft.fft2(tmp, axes=(0, 1))
                re, im = np.real(tmp_fft), np.imag(tmp_fft)
                im[-1-trigger_size:-1, -1-trigger_size:-1] = phase_trigger * np.pi
                tmp_fft = re + 1j * im
                tmp = np.fft.ifft2(tmp_fft, axes=(0, 1))
                tmp = tmp.real
                target[i:i+window_size, j:j+window_size, ch] = tmp
    return target

def zero_out_tensor(tensor, ratio):
    size = tensor.numel()
    num_zero = int(size * ratio)
    indices = torch.randperm(size)[:num_zero]
    tensor.reshape(-1)[indices] = 0
    return tensor

def get_artifact(config: DictConfig):
    scale = config.image_size
    if config.attack.name == config.artifact.name:
        artifacts = torch.zeros(size=(3, scale, scale))
        arti_config = config.copy()
        arti_config.attack = arti_config.artifact
        artifacts_p = patch_trigger(artifacts, arti_config)
    else:
        # e.g., attack ftrojan, artifact blend
        # print('hh')
        artifacts_1 = torch.zeros(size=(3, scale, scale))
        artifacts_2 = torch.zeros(size=(3, scale, scale))
        artifacts_1 = patch_trigger(artifacts_1, config=config)

        arti_config = config.copy()
        arti_config.attack = arti_config.artifact
        artifacts_2 = patch_trigger(artifacts_2, arti_config)

        artifacts_p = artifacts_1 + artifacts_2
        # torchvision.utils.save_image(artifacts_p.repeat(4, 1, 1, 1), f'../test/test_artifact.png', nrow=2)
    return artifacts_p
