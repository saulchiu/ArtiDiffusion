import torch
import torchvision
import PIL.Image

'''
20 male
'''
target = 11
ds = torchvision.datasets.CelebA(root='../data', split='train', transform=None, download=False)
l = []
for x, y in ds:
    y = 1 if y[20] == 1 else 0
    if y == 1:
        l.append(x)
    if len(l) > 10:
        break


for i, e in enumerate(l):
    e.save(f'../runs/celeba/{i}.png', 'PNG')