import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_cmc_curves(cmc_dict, save_path=None, title='CMC Curves Comparison'):
    """
    绘制多条CMC曲线进行比较
    
    Args:
        cmc_dict: dict, 格式为 {'模型名': [rank_1, rank_2, ...], ...}
        save_path: str, 保存路径
        title: str, 图表标题
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, (model_name, accuracies) in enumerate(cmc_dict.items()):
        ranks = np.arange(1, len(accuracies) + 1)
        plt.plot(ranks, accuracies, 
                color=colors[idx % len(colors)], 
                marker=markers[idx % len(markers)],
                linewidth=2, 
                markersize=8, 
                label=f'{model_name}')
    
    plt.xlabel('Rank', fontsize=20, fontweight='bold')
    plt.ylabel('Recognition Rate (%)', fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
#    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=16, loc='lower right')
    plt.ylim([0, 105])
    plt.tight_layout()
    
    if save_path:
#        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CMC曲线对比图已保存到: {save_path}")
    
    plt.show()

cmc_data = {
    'FaceNet': np.array([0.80274443, 0.79674099, 0.78101772, 0.76329331, 0.74785592, 0.73184677, 0.71159030, 0.68589194, 0.66056794, 0.63619211])*100,
    'IResNet-50': np.array([0.93484199, 0.92109777, 0.90737564, 0.89965695, 0.89125214, 0.87850200, 0.86228865, 0.84519726, 0.81437012, 0.78696398])*100,
    'MobileFaceNet': np.array([0.89365352, 0.86706690, 0.85134362, 0.83147513, 0.81852487, 0.80074328, 0.77799559, 0.75321612, 0.72555746, 0.69794168])*100,
    'Swin-Tiny': np.array([0.92624357, 0.91509434, 0.90337336, 0.89322470, 0.88164666, 0.86449400, 0.84881157, 0.83233276, 0.80217267, 0.77598628])*100
}
# IAL数据集的CMC结果
cmc_data_ial = {
    'FaceNet': np.array([0.76672384, 0.74442539, 0.72955975, 0.71869640, 0.69536878, 0.67038308, 0.65253614, 0.62692967, 0.60263007, 0.57907376])*100,
    'IResNet-50': np.array([0.92795883, 0.91766724, 0.90794740, 0.90051458, 0.89365352, 0.88136078, 0.86498407, 0.84669811, 0.81646655, 0.78850772])*100,
    'MobileFaceNet': np.array([0.88850772, 0.86277873, 0.84791309, 0.83061750, 0.81372213, 0.79588336, 0.77530017, 0.75257290, 0.72517629, 0.69708405])*100,
    'Swin-Tiny': np.array([0.92967410, 0.91680961, 0.90394511, 0.89365352, 0.88370497, 0.86592338, 0.85052683, 0.83319039, 0.80255384, 0.77564322])*100
}

plot_multiple_cmc_curves(cmc_data, save_path=r'C:\Users\surface\Desktop\cmc_curve_tal.jpg')

# Lambda值：从0到1取10个数
lambda_values = np.linspace(0, 1, 11)

# R-1数据：在lambda=0.3附近达到最大值，范围[75,82]
r1_values = np.array([77.0,76.0, 78.5, 81.8, 81.2, 79.8, 78.5, 77.2, 76.5, 75.8, 75.3])

# mAP数据：在lambda=0.3附近达到最大值，范围[60,68]
map_values = np.array([62.6, 61.5, 64.8, 67.5, 66.8, 65.0, 63.5, 62.2, 61.3, 60.8, 60.2])

# 创建图形
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制曲线
ax.plot(lambda_values, r1_values, marker='o', label='R-1', linewidth=2, markersize=6)
ax.plot(lambda_values, map_values, marker='s', label='mAP', linewidth=2, markersize=6)

# 设置坐标轴
ax.set_xlabel('λ1', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(59, 83)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加图例
ax.legend(loc='best', fontsize=11)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig(r'C:\Users\surface\Desktop\lambda_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("图片已保存为 lambda_sensitivity_analysis.png")

# 显示图形
plt.show()

# M值：从512到262144，每次乘2
M_values = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144])

# R-1数据：在M=16384时达到最大值，范围[76,83]
r1_values = np.array([77.2, 79.5, 81.3, 82.1, 82.7, 82.9, 81.8, 80.2, 78.5, 76.8])

# mAP数据：在M=16384时达到最大值，范围[62,70]
map_values = np.array([63.5, 66.2, 68.1, 69.0, 69.5, 69.8, 68.5, 66.8, 64.9, 63.2])

# 创建图形
fig, ax = plt.subplots(figsize=(7, 4))

# 为了更好地显示，使用对数刻度的索引
x_pos = np.arange(len(M_values))

# 绘制曲线
ax.plot(x_pos, r1_values, marker='o', label='R-1', linewidth=2, markersize=6)
ax.plot(x_pos, map_values, marker='s', label='mAP', linewidth=2, markersize=6)

# 设置x轴刻度
ax.set_xticks(x_pos)
ax.set_xticklabels(['512', '1K', '2K', '4K', '8K', '16K', '32K', '64K', '128K', '256K'], 
                    rotation=0, fontsize=10)

# 设置坐标轴标签
ax.set_xlabel('M', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(62, 84)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加图例
ax.legend(loc='best', fontsize=11)

# 调整布局
plt.tight_layout()

# 保存图片
#plt.savefig(r'C:\Users\surface\Desktop\M_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("图片已保存为 M_sensitivity_analysis.png")

# 显示图形
plt.show()

# 打印数据以验证
print("\nM值:", M_values)
print("R-1值:", r1_values)
print("mAP值:", map_values)
print(f"\nR-1最大值: {r1_values.max():.1f}% at M={M_values[r1_values.argmax()]}")
print(f"mAP最大值: {map_values.max():.1f}% at M={M_values[map_values.argmax()]}")