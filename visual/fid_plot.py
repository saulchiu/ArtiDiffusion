import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# from matplotlib import font_manager
# for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
#     print(font)


# 设置字体为 Calibri
rcParams['font.family'] = 'Calibri'

# 数据准备
poison_rates = [0, 5, 10, 30, 50, 70]
datasets = [
    "GTSRB+BadNet",
    "GTSRB+Blended",
    "CIFAR10+BadNet",
    "CIFAR10+Blended",
    "CelebA+BadNet",
    "CelebA+Blended",
]
values = [
    [7.41, 8.6, 7.97, 15.62, 15.63, 34.51],  # GTSRB+BadNet
    [7.41, 11.63, 6.01, 12.38, 25.87, 49.55],  # GTSRB+Blended
    [5.75, 5.69, 11.97, 7.42, 19.15, 20.15],  # CIFAR10+BadNet
    [5.75, 8.87, 10.1, 12.84, 27.45, 29.2],  # CIFAR10+Blended
    [10.88, 17.81, 21.58, 26.37, 25.87, 29.2],  # CelebA+BadNet
    [8.88, 7.97, 8.25, 12.38, 19.15, 16.71],  # CelebA+Blended
]

# colors = ["#0d5b26", "#c94733", "#fddf8b", "#3fab47", "#52b9d8", "#2e5fa1"]
colors = ["#66c2a5", "#fc8d62", "#8da0cb", 
          "#e78ac3", "#a6d854", "#ffd92f", 
          "#e5c494", "#b3b3b3"]
colors = ["#648fff", "#785ef0", "#dc267f", 
          "#fe6100", "#ffb000", "#000000"]

colors = ["#648fff", "#ff7f0e", "#a6d854", "#d62728", 
          "#e78ac3", "#17becf"]

x = np.arange(len(poison_rates))  # x轴刻度
bar_width = 0.15  # 柱状图宽度
spacing = 0.00  # 柱子之间的间隔

# 图表绘制
fig, ax = plt.subplots(figsize=(14, 6))

# 每组数据绘制
for i, (dataset, color) in enumerate(zip(datasets, colors)):
    ax.bar(
        x + i * (bar_width + spacing),
        values[i],
        bar_width,
        label=dataset,
        color=color,
        alpha=0.8,
    )

# 添加数值标签
for i in range(len(datasets)):
    for j in range(len(poison_rates)):
        ax.text(
            x[j] + i * (bar_width + spacing),
            values[i][j] + 0.5,
            str(values[i][j]),
            ha="center",
            va="bottom",
            fontsize=8,
        )

# 添加标签和标题
ax.set_xlabel("Poisoning Rate")
ax.set_ylabel("FID")
ax.set_title(
    "FID of Generated Sample vs. Poison Rate (GTSRB, CIFAR10, CelebA, Trigger: BadNet, Blended)"
)
ax.set_xticks(x + (len(datasets) - 1) * (bar_width + spacing) / 2)
ax.set_xticklabels(poison_rates)
ax.legend(loc="upper left")  # 将图例移动到左上角

# 调整坐标轴
ax.set_xlim(-bar_width - spacing, x[-1] + len(datasets) * (bar_width + spacing))
ax.spines["left"].set_position(("data", -bar_width - spacing))

# 展示图表
plt.tight_layout()
plt.show()

plt.savefig('chart.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("图表已保存为 chart.pdf")
