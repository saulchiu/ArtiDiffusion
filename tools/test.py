import time

import PIL.Image
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms

def test(e):
    image_np = e.cpu().detach().numpy()
    image_np = image_np.transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    image = PIL.Image.fromarray(image_np)
    image.save(f'{time.time_ns()}.png')

trigger = PIL.Image.open('../resource/blended/hello_kitty.jpeg')
x_start = PIL.Image.open('../dataset/dataset-cifar10-good/good_0.png')
x_start_p = PIL.Image.open('1716119158227650584.png')
trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32, 32))
])
x_start_p = trans(x_start_p)
x_start = trans(x_start)
trigger = trans(trigger)
g_p = (x_start_p - 0.8 * x_start) / 0.2
test(trigger)



