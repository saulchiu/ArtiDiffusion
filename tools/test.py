import os

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for i in range(1000):
    writer.add_scalar("loss", 2 * i, i)
    writer.flush()
