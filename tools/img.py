import torchvision
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def save_one(tensor, path):
    image_np = tensor.cpu().detach().numpy()
    image_np = image_np.transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)
    image.save(path)


def cal_ssim(tensor_1, tensor_2, data_range=1):
    tensor_1 = tensor_1.to('cuda')
    tensor_2 = tensor_2.to('cuda')
    n_1 = tensor_1.cpu().detach().numpy()
    n_2 = tensor_2.cpu().detach().numpy()
    n_1 = n_1.astype(np.float64)
    n_2 = n_2.astype(np.float64)
    return ssim(n_1, n_2, multichannel=True, win_size=3, data_range=data_range)


if __name__ == "__main__":
    triger_path = '../resource/badnet/trigger_image_grid.png'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32))
    ])
    triger = Image.open(triger_path)
    triger = transform(triger).to('cuda:0')
    print(cal_ssim(triger, triger))
