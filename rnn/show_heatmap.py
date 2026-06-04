import torch
import matplotlib.pyplot as plt

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """显示矩阵热图"""
    # 获取矩阵的行数和列数（对应输入张量的前两个维度）
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    
    # 创建子图网格，squeeze=False 确保 axes 始终是二维数组，方便后续遍历
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    
    # 遍历每一行和每一列来绘制热图
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # detach().numpy() 将 PyTorch 张量转换为 NumPy 数组，因为 matplotlib 只认 NumPy
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            
            # 只在最底下一行加上 X 轴标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 只在最左边一列加上 Y 轴标签
            if j == 0:
                ax.set_ylabel(ylabel)
            # 如果有标题，则设置子图标题
            if titles:
                ax.set_title(titles[j])
                
    # 在所有子图旁边添加一个颜色条 (colorbar)
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    
    # 显式展示图像
    plt.show()

if __name__ == '__main__':

    # --- 测试代码 ---
    # 创建一个 10x10 的单位矩阵，并改变形状为 (1, 1, 10, 10)
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')