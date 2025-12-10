import matplotlib.pyplot as plt

# =======================
# 示例数据（请自行替换）
# =======================
epochs = [0, 250, 500, 1000, 2000]

asr_1 = [74.41, 67.19, 0, 0, 0]          
asr_3 = [83.59, 72.66, 39.06, 0, 0]      
asr_5 = [84.77, 49.22, 24.22, 0, 0]        

# =======================
# 开始画图
# =======================
plt.figure(figsize=(12, 4))

plt.plot(epochs, asr_1, marker='o', color='royalblue', label='Poison Rate 0.1')
plt.plot(epochs, asr_3, marker='^', color='limegreen', label='Poison Rate 0.3')
plt.plot(epochs, asr_5, marker='D', color='orange', label='Poison Rate 0.5')

plt.xlabel('epoch')
plt.ylabel('ASR')
plt.title('ASR of ANP v.s. train epoch')
plt.grid(True)

plt.legend()

plt.tight_layout()
plt.show()

plt.savefig('chart.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("图表已保存为 chart.pdf")
