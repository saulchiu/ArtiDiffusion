import PIL.Image
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.transforms import ToTensor, Resize, Compose

img = Image.open('./hello_kitty.jpeg')
scale = 64
trans = Compose([ToTensor(), Resize((scale, scale))])
img_tensor = trans(img)
img_np = img_tensor.detach().numpy()
img_np = img_np.transpose((1, 2, 0))
img_np = (img_np * 255.).astype(np.uint8)
img = PIL.Image.fromarray(img_np)
img.save(f'./hello_kitty_{scale}.jpg')

