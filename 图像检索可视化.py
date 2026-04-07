import matplotlib.pyplot as plt
import numpy as np

def visualize_retrieval_results(query_image, retrieval_images, retrieval_scores=None, retrieval_labels=None, query_label=None):
    """
    可视化图像检索结果
    """
    # 创建画布，调整大小和布局
    fig = plt.figure(figsize=(16, 12))
    
    # 使用GridSpec创建清晰的布局：4行5列 + 查询图像单独区域
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 0.3, 1, 1], 
                         hspace=0.5, wspace=0.3)
    
    # 1. 显示查询图像（居中显示，占用中间3列）
    ax_query = fig.add_subplot(gs[0, 1:4])
    ax_query.imshow(query_image)
    
    # 构建查询图像标题
    ax_query.set_title('Query Image', fontsize=18, fontweight='bold', pad=20)
    
    # 如果有查询标签，单独添加（不加粗）
    if query_label is not None:
        ax_query.text(0.5, -0.1, f'Label: {query_label}', 
                     ha='center', va='top', fontsize=14, 
                     transform=ax_query.transAxes)
    
    ax_query.axis('off')
    
    # 2. 添加标题（跨越所有列居中）
    ax_title = fig.add_subplot(gs[1, :])
    ax_title.text(0.5, 0.5, 'Retrieval Results (Top-10)', 
                  ha='center', va='center', fontsize=16, fontweight='bold',
                  transform=ax_title.transAxes)
    ax_title.axis('off')
    
    # 3. 显示检索结果（2行，每行5张图）
    for i in range(min(10, len(retrieval_images))):
        row = 2 + (i // 5)  # 第2行或第3行
        col = i % 5         # 列位置0-4
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(retrieval_images[i])
        
        # 构建标题
        title_parts = [f'Top-{i+1}']
        if retrieval_scores is not None:
            title_parts.append(f'Score: {retrieval_scores[i]:.3f}')
        if retrieval_labels is not None:
            title_parts.append(f'Label: {retrieval_labels[i]}')
            
        ax.set_title('\n'.join(title_parts), fontsize=10, pad=8)
        ax.axis('off')
    
    # 4. 如果检索图像不足10张，隐藏多余的子图
    for i in range(len(retrieval_images), 10):
        row = 2 + (i // 5)
        col = i % 5
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
query_img = np.random.rand(128, 128, 3)
retrieval_imgs = [np.random.rand(64, 64, 3) for _ in range(10)]
scores = [0.95 - i*0.08 for i in range(10)]
labels = [f'Class {i%3}' for i in range(10)]
query_label = 'Class A'

visualize_retrieval_results(query_img, retrieval_imgs, scores, labels, query_label)