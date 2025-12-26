import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和风格，使其看起来更清新
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # 兼顾英文和中文显示
plt.rcParams['axes.unicode_minus'] = False

def draw_cute_chart(ax, data_blue, data_red, title_text):
    """
    绘制单个可爱风格图表的函数
    """
    # X轴标签
    categories = ["No-Reasoning\n(L=0)", "Reasoning\n(L=2048)", "Reasoning\n(L=4096)"]
    x = np.arange(len(categories))

    # === 定义配色 (Cute Palette) ===
    # 柔和的深蓝与珊瑚红，比纯色更耐看
    color_blue = '#4A90E2'  # 鲜亮的柔和蓝
    color_red = '#FF6B6B'   # 温暖的珊瑚红
    bg_color = '#Fdfdfd'    # 极淡的灰白背景，不刺眼

    ax.set_facecolor(bg_color)
    
    # === 绘制线条 ===
    # Perturbation-II (Red, Dashed)
    ax.plot(x, data_red, color=color_red, linestyle='--', linewidth=2.5, 
            marker='*', markersize=14, label='Perturbation-II', zorder=10)
    
    # Perturbation-I (Blue, Solid)
    ax.plot(x, data_blue, color=color_blue, linestyle='-', linewidth=2.5, 
            marker='o', markersize=10, markeredgewidth=2, markeredgecolor='white', label='Perturbation-I', zorder=11)

    # === 添加圆角数值标签 (Cute Touch) ===
    # 蓝色标签
    for i, val in enumerate(data_blue):
        ax.annotate(f'{val}%', xy=(x[i], val), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, color=color_blue, fontweight='bold',
                    bbox=dict(boxstyle="round4,pad=0.4", fc="white", ec=color_blue, lw=1.5, alpha=0.9))

    # 红色标签
    for i, val in enumerate(data_red):
        ax.annotate(f'{val}%', xy=(x[i], val), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, color=color_red, fontweight='bold',
                    bbox=dict(boxstyle="round4,pad=0.4", fc="white", ec=color_red, lw=1.5, alpha=0.9))

    # === 轴设置 ===
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='medium', color='#555555')
    
    # Y轴范围稍微留白
    ax.set_ylim(0, 48)
    ax.set_ylabel("Alignment Loss Rate ($\Delta\%$)", fontsize=12, fontweight='bold', color='#333333')
    ax.set_xlabel("Reasoning Depth ($L$)", fontsize=12, fontweight='bold', color='#333333')
    
    # 格式化Y轴为百分比
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
    
    # === 美化网格 ===
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#aaaaaa', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    # 图例
    legend = ax.legend(loc='upper left', frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color("#555555")

# === 准备数据 ===
# 图1的数据
img1_blue = [10.0, 17.0, 21.1]
img1_red  = [32.0, 37.3, 39.5]

# 图2的数据
img2_blue = [6.6, 11.5, 11.9]
img2_red  = [9.2, 20.8, 33.1]

# === 创建画布 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# 绘制两张图
draw_cute_chart(ax1, img1_blue, img1_red, "Chart 1")
draw_cute_chart(ax2, img2_blue, img2_red, "Chart 2")

plt.tight_layout(pad=3.0)
plt.show()