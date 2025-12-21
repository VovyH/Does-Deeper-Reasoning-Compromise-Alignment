import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 加载数据
with open(r'D:\个人信息\Code\2025ACL_CHJ\2025ACL_CHJ\CHJ\hallucination_jailbeak\data\attention_trace_full_input.json', 'r') as f:
    simple_records = json.load(f)

# 提取原始数据
steps = range(len(simple_records))
input_attns = [r["input_attn"] for r in simple_records]
thinking_attns = [r["thinking_attn"] for r in simple_records]

# 数据处理：每5步一个窗口，计算均值、最小、最大
window = 5
smoothed_steps = []
smoothed_input_mean = []
smoothed_input_min = []
smoothed_input_max = []
smoothed_thinking_mean = []
smoothed_thinking_min = []
smoothed_thinking_max = []

for i in range(0, len(simple_records), window):
    window_input = input_attns[i:i+window]
    window_thinking = thinking_attns[i:i+window]
    
    if window_input:  # 避免最后一个不完整的窗口为空
        smoothed_steps.append(i)  # 用窗口起始步数作为x坐标
        smoothed_input_mean.append(np.mean(window_input))
        smoothed_input_min.append(np.min(window_input))
        smoothed_input_max.append(np.max(window_input))
        
        smoothed_thinking_mean.append(np.mean(window_thinking))
        smoothed_thinking_min.append(np.min(window_thinking))
        smoothed_thinking_max.append(np.max(window_thinking))

# 设置绘图风格
plt.style.use('seaborn-v0_8-pastel')

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制主曲线 + 浅色震荡带
# Input (蓝色)
ax.plot(smoothed_steps, smoothed_input_mean, 
        color='#4a7abc', label='Attention to Input', 
        linewidth=2.5, marker='o', markersize=5)
ax.fill_between(smoothed_steps, smoothed_input_min, smoothed_input_max, 
                color='#4a7abc', alpha=0.15, label='_nolegend_')  # 很浅的填充

# Thinking (红色)
ax.plot(smoothed_steps, smoothed_thinking_mean, 
        color='#e66465', label='Attention to Thinking Process', 
        linewidth=2.5, marker='s', markersize=5)
ax.fill_between(smoothed_steps, smoothed_thinking_min, smoothed_thinking_max, 
                color='#e66465', alpha=0.15, label='_nolegend_')

# 标签和设置
ax.set_xlabel('Thinking Depth', fontsize=12, labelpad=10)
ax.set_ylabel('Total Attention Weight (Sum)', fontsize=12, labelpad=10)

ax.set_ylim(0, 1.1)
ax.set_xlim(0, len(steps))

ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(fontsize=10, loc='best', frameon=True, framealpha=0.8)

for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.5)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.show()