import torch
import torch.nn as nn

# 模型输出和目标值
y_p = torch.tensor([[-0.1709, 0.8022, 1.3744, 0.211, 0.2323, -0.122, 0.111, 0.3343, -0.5338, 1.4061],
                    [0.0728, 0.4386, 1.9230, 0.213, 0.121, -0.111, 0.111, 0.3911, -0.5742, 2.0375]],
                   device='cuda:0', requires_grad=True)

y = torch.tensor([8, 0], device='cuda:0')

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(y_p, y)
print(loss)
