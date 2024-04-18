import numpy as np
from PIL import Image


def save(tensor, path):
    image_np = tensor.cpu().detach().numpy()
    image_np = image_np.transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)
    image.save(path)