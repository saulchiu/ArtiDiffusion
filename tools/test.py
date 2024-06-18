import torch

t1 = torch.randn(size=(128, 3, 32, 32), device="cuda:0")
ts = [t1, t1]
# ts: list -> tensor (256, 3, 32, 32)
ts = torch.cat(ts, dim=0)
print(ts.shape)