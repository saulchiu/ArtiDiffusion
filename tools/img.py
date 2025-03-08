import cv2
import numpy
import numpy as np
import pywt
import torch
from matplotlib import pyplot as plt
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def ndarray2tensor(nd: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor.
    :param nd: a numpy array, e.g., shape (32, 32, 3)
    :return: a torch tensor, e.g., with shape (3, 32, 32)
    """
    # Swap the axes so that channels are first, from (H, W, C) to (C, H, W)
    tensor = (torch.from_numpy(nd) / 255.).permute(2, 0, 1)
    return tensor


def tensor2ndarray(t: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    :param t: a tensor, e.g., with shape (3, 32, 32)
    :return: a numpy array, e.g., with shape (32, 32, 3)
    """
    # Permute the tensor to swap back to (H, W, C) from (C, H, W)
    nd = (t * 255.).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    return nd


def dct_2d_3c_slide_window(x_train: np.ndarray, window_size=32):
    """
    input should be floated
    :param x_train:
    :param window_size:
    :return:
    """
    x_train = np.transpose(x_train, (2, 0, 1))
    x_dct = np.zeros_like(x_train, dtype=float)
    for ch in range(x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                x_dct[ch][w:w + window_size, h:h + window_size] = dct(
                    dct((x_train[ch][w:w + window_size, h:h + window_size].astype(float)).T, norm='ortho').T,
                    norm='ortho')
    x_dct = np.transpose(x_dct, (1, 2, 0))
    return x_dct


def idct_2d_3c_slide_window(x_train: np.ndarray, window_size=32):
    """
    input should be floated
    :param x_train:
    :param window_size:
    :return:
    """
    x_train = np.transpose(x_train, (2, 0, 1))
    x_idct = np.zeros(x_train.shape, dtype=float)
    for ch in range(0, x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                x_idct[ch][w:w + window_size, h:h + window_size] = idct(
                    idct((x_train[ch][w:w + window_size, h:h + window_size].astype(float)).T, norm='ortho').T,
                    norm='ortho')
    x_idct = np.transpose(x_idct, (1, 2, 0))
    return x_idct


def dct_2d_3c_full_scale(x: np.ndarray):
    return dct_2d_3c_slide_window(x, x.shape[0])

def idct_2d_3c_full_scale(x: np.ndarray):
    return idct_2d_3c_slide_window(x, x.shape[0])


def dwt_2d_3c(img: np.ndarray, wavelet='haar', clip=False) -> list:
    """

    :param clip:
    :param wavelet:
    :param img: ndarray, shape (w, h, c), element should between 0 and 1
    :return: [LL, LH, HL, HH] between 0 and 1
    """
    LL_r, (LH_r, HL_r, HH_r) = pywt.dwt2(img[:, :, 0], wavelet)
    LL_g, (LH_g, HL_g, HH_g) = pywt.dwt2(img[:, :, 1], wavelet)
    LL_b, (LH_b, HL_b, HH_b) = pywt.dwt2(img[:, :, 2], wavelet)

    LL = np.zeros(shape=(LL_r.shape[0], LL_r.shape[1], 3))
    LH = np.zeros(shape=(LH_r.shape[0], LH_r.shape[1], 3))
    HL = np.zeros(shape=(HL_r.shape[0], HL_r.shape[1], 3))
    HH = np.zeros(shape=(HH_r.shape[0], HH_r.shape[1], 3))

    LL[:, :, 0], LL[:, :, 1], LL[:, :, 2] = LL_r, LL_g, LL_b
    LH[:, :, 0], LH[:, :, 1], LH[:, :, 2] = LH_r, LH_g, LH_b
    HL[:, :, 0], HL[:, :, 1], HL[:, :, 2] = HL_r, HL_g, HL_b
    HH[:, :, 0], HH[:, :, 1], HH[:, :, 2] = HH_r, HH_g, HH_b

    if clip:
        LL = np.clip(LL, 0., 1.)
        LH = np.clip(LH, 0., 1.)
        LH = np.clip(LH, 0., 1.)
        HH = np.clip(HH, 0., 1.)
    return [LL, (LH, HL, HH)]


def idwt_2d_3c(coeffs: list, wavelet='haar') -> np.ndarray:
    """
    Perform the inverse discrete wavelet transform (IDWT) for a 3-channel RGB image.

    :param coeffs: list of [LL, LH, HL, HH], each of shape (w, h, 3)
    :param wavelet: wavelet type, default is 'haar'
    :return: reconstructed image as ndarray with shape (w, h, 3), element values between 0 and 1
    """
    LL, (LH, HL, HH) = coeffs

    LL_r, LH_r, HL_r, HH_r = LL[:, :, 0], LH[:, :, 0], HL[:, :, 0], HH[:, :, 0]
    LL_g, LH_g, HL_g, HH_g = LL[:, :, 1], LH[:, :, 1], HL[:, :, 1], HH[:, :, 1]
    LL_b, LH_b, HL_b, HH_b = LL[:, :, 2], LH[:, :, 2], HL[:, :, 2], HH[:, :, 2]

    img_r = pywt.idwt2((LL_r, (LH_r, HL_r, HH_r)), wavelet)
    img_g = pywt.idwt2((LL_g, (LH_g, HL_g, HH_g)), wavelet)
    img_b = pywt.idwt2((LL_b, (LH_b, HL_b, HH_b)), wavelet)

    img_reconstructed = np.stack((img_r, img_g, img_b), axis=-1)
    img_reconstructed = np.clip(img_reconstructed, 0., 1.)

    return img_reconstructed

def fft_2d_3c(x_0: np.ndarray):
    x_f = []
    for i in range(3):
        x_f.append(np.fft.fft2(x_0[:, :, i]))
    x_f = np.stack(x_f, axis=2)
    return x_f

def ifft_2d_3c(x_0: np.ndarray):
    x_i = []
    for i in range(3):
        x_i.append(np.fft.ifft2(x_0[:, :, i]))
    x_i = np.stack(x_i, axis=2)
    return x_i


def rgb_to_yuv(R,G,B):
    """
    Converts an RGB image to YUV using the BT.601 standard.
    watch out: 
    1. R, G, B shold in (0, 255), float or integer are alright.
    2. Y, U, V need CLIP operation!
    """
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (B - Y) * 0.492 + 128
    V = (R - Y) * 0.877 + 128
    return Y,U,V

def yuv_to_rgb(Y, U, V):
    """
    Converts a YUV image back to RGB using the BT.601 standard.
    """
    R = Y + 1.140 * (V - 128)
    G = Y - 0.394 * (U - 128) - 0.581 * (V - 128)
    B = Y + 2.032 * (U - 128)
    
    return R, G, B

def rgb2yuv(x_rgb: numpy.ndarray) -> numpy.ndarray:
    """
    this function converts RGB image to YUV image both a single image and a batch
    :param x_rgb:
    :return:
    """
    if len(x_rgb.shape) == 4:  # batch
        x_yuv = np.zeros(x_rgb.shape, dtype=float)
        for i in range(x_rgb.shape[0]):
            img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YUV)
            x_yuv[i] = img
        return x_yuv
    return cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YUV)


def yuv2rgb(x_yuv):
    """
    this function converts YUV image to RGB image both a single image and a batch
    :param x_yuv:
    :return:
    """
    if len(x_yuv.shape) == 4:  # batch
        x_rgb = np.zeros(x_yuv.shape, dtype=float)
        for i in range(x_yuv.shape[0]):
            img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YUV2RGB)
            x_rgb[i] = img
        return x_rgb
    return cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YUV2RGB)


from skimage.color import lab2rgb, rgb2lab


from skimage.color import rgb2lab
import torch

def rgb_tensor_to_lab_tensor(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB tensor to a LAB tensor.
    
    Args:
        rgb_tensor (torch.Tensor): Input tensor in RGB format with shape (B, C, H, W) and values normalized to [0, 1].
    
    Returns:
        torch.Tensor: Output tensor in LAB format with shape (B, C, H, W) and values normalized to [0, 1].
    """
    # Ensure the tensor is in the range [0, 1]
    # if rgb_tensor.max() > 1.0 or rgb_tensor.min() < 0.0:
    #     raise ValueError("Input tensor values must be in the range [0, 1].")
    
    # Convert to numpy array with shape (B, H, W, C)
    rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C), float32, values in [0, 1]
    
    # Convert RGB to LAB using skimage.color.rgb2lab
    lab_np = rgb2lab(rgb_np)  # (B, H, W, C), float64, L in [0, 100], a and b in [-128, 127]
    
    # Normalize LAB values to [0, 1]
    lab_np[:, :, :, 0] /= 100.0  # L channel: [0, 100] -> [0, 1]
    lab_np[:, :, :, 1:] = (lab_np[:, :, :, 1:] + 128.0) / 255.0  # a and b channels: [-128, 127] -> [0, 1]
    
    # Convert back to PyTorch tensor with shape (B, C, H, W)
    lab_tensor = torch.from_numpy(lab_np.astype('float32')).permute(0, 3, 1, 2).contiguous()
    
    return lab_tensor.float()

def lab_tensor_to_rgb_tensor(lab_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a LAB tensor to an RGB tensor.
    
    Args:
        lab_tensor (torch.Tensor): Input tensor in LAB format with shape (B, C, H, W) and values normalized to [0, 1].
    
    Returns:
        torch.Tensor: Output tensor in RGB format with shape (B, C, H, W) and values in [0, 1].
    """
    # Ensure the tensor is in the range [0, 1]
    # if lab_tensor.max() > 1.0 or lab_tensor.min() < 0.0:
    #     raise ValueError("Input tensor values must be in the range [0, 1].")
    
    # Convert LAB values back to their original ranges
    lab_tensor = lab_tensor.clone()  # Avoid modifying the original tensor
    lab_tensor[:, 0] *= 100.0  # L channel: [0, 1] -> [0, 100]
    lab_tensor[:, 1:] = lab_tensor[:, 1:] * 255.0 - 128.0  # a and b channels: [0, 1] -> [-128, 127]

    # Convert to numpy array with shape (B, H, W, C)
    lab_np = lab_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    
    # Convert LAB to RGB using skimage.color.lab2rgb
    rgb_np = lab2rgb(lab_np)  # (B, H, W, C), float32, values in [0, 1]
    
    # Convert back to PyTorch tensor with shape (B, C, H, W)
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).contiguous()
    
    return rgb_tensor.float()

def split_lab_channels(image):
    assert isinstance(image, torch.Tensor)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return torch.split(image, [1, 2], dim=1)


def clip(data: numpy.ndarray) -> numpy.ndarray:
    if data.shape[0] > 64:
        return np.clip(a=data, a_min=1.5, a_max=4.5)
    else:
        from scipy.ndimage import gaussian_filter
        data = np.log1p(np.abs(data))
        data = gaussian_filter(data, sigma=2)
        return data


def plot_space_target_space(x_space: numpy.ndarray, x_target, x_process_space, x_process_target, is_clip: bool = False):
    """
    compare the image in original domain (e.g., space domain) and target domain (e.g., frequency domain)
    :param x_space:
    :param x_target: the image in target domain after target transform
    :param x_process_space:
    :param x_process_target: the processed image (e.g., poisoning image) in target domain after target transform
    :param is_clip: choose whether clip the value in target domain
    :return:
    """
    if is_clip:
        x_target = clip(x_target)
        x_process_target = clip(x_process_target)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(x_space)
    axs[0, 0].set_title(f'Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(x_target[:, :, 0], cmap='hot')
    axs[0, 1].set_title('Original Image DCT')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(x_process_space)
    axs[1, 0].set_title(f'after (attack) process')
    axs[1, 0].axis('off')

    im2 = axs[1, 1].imshow(x_process_target[:, :, 0], cmap='hot')
    axs[1, 1].set_title(f'(attacked) img in target space')
    axs[1, 1].axis('off')
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax)
    plt.tight_layout()
    plt.show()
    

def get_shifted_amp_pha(x_spatial):
    x_fft = np.fft.fft2(x_spatial, axes=(-3, -2))
    amp = np.abs(x_fft)
    pha = np.angle(x_fft)
    amp_shift = np.fft.fftshift(amp, axes=(-3, -2))
    pha_shift = np.fft.fftshift(pha, axes=(-3, -2))
    return amp_shift, pha_shift






