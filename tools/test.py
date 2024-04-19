import torch

if __name__ == '__main__':
    loss = torch.tensor([1., 2, 3])
    ssim = 0.5
    loss += ssim
    print(loss.mean())