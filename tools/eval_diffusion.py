import math

import PIL.Image
import detectors
import timm
import torch
from denoising_diffusion_pytorch import Unet, Trainer
from PIL import Image
import sys
import torchvision.transforms.transforms as T

sys.path.append('../')
from backdoor_diffusion.badnet_diffusion import BadDiffusion
import torchvision.transforms
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset, GaussianDiffusion
import matplotlib.pyplot as plt
import torch.utils
from tools.img import cal_ssim
from models.resnet import ResNet18
from torch.utils.data import DataLoader

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_cifar10(net, x):
    model_out = net(x.detach().reshape(1, 3, 32, 32))
    probs = torch.nn.functional.softmax(model_out, dim=1)
    # 找到概率最大的索引
    max_prob_index = probs.argmax(dim=1).item()  # 使用 .item() 来获取标量值
    # 将索引映射到 CIFAR-10 的类别
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = class_names[max_prob_index]
    print(predicted_class)
    return predicted_class


def plot_images(images, num_images, net=None):
    cols = int(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    figsize_width = cols * 5
    figsize_height = rows * 5
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height))
    axes = axes.flatten()  # Flatten the array for easier iteration

    # Plot each image on the grid
    for idx, (img, ax) in enumerate(zip(images, axes)):
        if idx < num_images:  # Only plot the actual number of images
            img_ssim = cal_ssim(img, images[0])
            if net is not None:
                label_p = predict_cifar10(net, img)
            ax.imshow(img.permute(1, 2, 0).cpu().numpy())
            ax.axis('off')
            # Add SSIM value below the image
            ax.text(0.5, -0.08, f'SSIM: {img_ssim:.2f}', transform=ax.transAxes, ha='center',
                    fontsize=10)
        else:
            ax.axis('off')  # Turn off the last empty subplot

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def load_bad_diffusion(path, device='cuda:0'):
    ld = torch.load(path)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    diffusion = BadDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        objective='pred_x0',
        trigger=None,
        loss_mode=4,
        factor_list=[1, 1 ,1],
        device='cuda:0'
    )
    diffusion.load_state_dict(ld['model'])
    diffusion = diffusion.to(device)
    return diffusion


def load_diffusion(path, device='cuda:0'):
    ld = torch.load(path)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=250
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )
    diffusion.load_state_dict(ld['model'])
    diffusion = diffusion.to(device=device)
    return diffusion


def sample_from_diffusion(diff_path, batch=4):
    with torch.inference_mode():
        diffusion = load_bad_diffusion(diff_path)
        sampled_images = diffusion.sample(batch_size=batch)
        # utils.save_image(sampled_images, fp='./test.png', nrow=4)
        plot_images(images=sampled_images, num_images=batch)


def sample_and_reconstruct(diffusion, x_start, t=10, device='cuda:0', plot=False):
    tensor_list = [x_start]
    x_noice = diffusion.q_sample(x_start=x_start, t=torch.tensor([t]).to(device))
    tensor_list.append(x_noice)
    x_noice = x_noice.reshape(1, 3, 32, 32)
    _, x_s = diffusion.p_sample(x=x_noice, t=t)
    x_s = x_s.reshape(3, 32, 32)
    tensor_list.append(x_s)
    tensors = torch.stack(tensor_list, dim=0)
    if plot:
        plot_images(images=tensors, num_images=len(tensor_list))
    return x_s


def sample_and_reconstruct_loop(diffusion, net, x_start, t=10, device='cuda:0', plot=False, loop=5):
    tensor_list = [x_start]
    for i in range(loop):
        x_t1 = sample_and_reconstruct(diffusion, x_start, t)
        tensor_list.append(x_t1)
        x_start = x_t1
    tensors = torch.stack(tensor_list, dim=0)
    plot_images(images=tensors, num_images=tensors.shape[0], net=net)


def laod_badnet(path, device='cuda:0'):
    ld = torch.load(path, map_location=device)
    net = ResNet18(num_classes=10).to(device=device)
    net.load_state_dict(ld)
    return net


if __name__ == '__main__':
    device = 'cuda:0'
    t = 25
    loop = 8
    ld = torch.load('../models/checkpoint/attack_result.pt')
    from models.preact_resnet import PreActResNet18
    net = PreActResNet18()
    net.load_state_dict(ld['model'])
    net = net.to(device)
    diffusion = load_bad_diffusion('../backdoor_diffusion/res_badnet_grid_cifar10_step10k_ratio1_loss5_factor1/model-10.pt',
                               device=device)
    # x_start = Image.open('../dataset/dataset-cifar10-badnet-trigger_image_grid/bad_8.png')
    trigger = PIL.Image.open('../resource/badnet/trigger_image_grid.png')
    mask = PIL.Image.open('../resource/badnet/trigger_image.png')
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    trigger = trans(trigger)
    mask = trans(mask)
    normal_data = torchvision.datasets.CIFAR10(
        root='../data/', train=False, transform=trans, download=False
    )
    normal_loader = torch.utils.data.DataLoader(dataset=normal_data, batch_size=128, shuffle=True, num_workers=1)
    x_start, index = next(iter(normal_loader))
    x_start = x_start[0]
    index = index[0]
    x_start = x_start.reshape(3, 32, 32)
    # add trigger
    x_start = (1 - mask) * x_start + mask * trigger
    x_start = x_start.to(device)
    print(f'real label is: {class_names[int(index)]}')
    sample_and_reconstruct_loop(diffusion, net, x_start, t, device, False, loop)

