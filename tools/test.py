import torch
import torch.nn as nn
import numpy as np

ld = np.load(
    '../backdoor_diffusion/res_bad_dataset_error/res_badnet_grid_cifar10_step10k_ratio1_loss5/factor2/dataset_stats.npz')
print()