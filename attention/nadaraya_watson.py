import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 0. 辅助工具函数：画热图
# ==========================================
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(3.5, 3.5), cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1: ax.set_xlabel(xlabel)
            if j == 0: ax.set_ylabel(ylabel)
            if titles: ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()

# ==========================================
# 1. 数据生成
# ==========================================
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本加上噪声
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实无噪声输出
n_test = len(x_test)  # 测试样本数

# 自定义绘图函数（替代 d2l.plot）
def plot_kernel_reg(y_hat, title):
    plt.figure(figsize=(6, 4))
    plt.plot(x_test, y_truth, label='Truth', color='black', linestyle='-')
    plt.plot(x_test, y_hat, label='Pred', color='blue', linestyle='--')
    plt.scatter(x_train, y_train, label='Train Data', color='red', marker='o', alpha=0.5)
    plt.xlim([0, 5])
    plt.ylim([-1, 5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# ==========================================
# 2. 非参数注意力机制 (Non-parametric Attention)
# ==========================================
print(">>> 运行非参数注意力核回归...")
# X_repeat 形状: (n_test, n_train)
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))

# attention_weights 形状：(n_test, n_train)
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)

# y_hat的每个元素都是值的加权平均值
y_hat = torch.matmul(attention_weights, y_train)

# 绘制拟合图与热图
plot_kernel_reg(y_hat, title="Non-parametric Attention")
show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
              xlabel='Sorted training inputs',
              ylabel='Sorted testing inputs')

# ==========================================
# 3. 带参数注意力机制 (Parametric Attention)
# ==========================================
print(">>> 运行带参数注意力核回归...")

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化一个可学习的标量参数 w
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries 转换形状以适配计算
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # 乘上可学习的参数 w
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # 使用 bmm (Batch Matrix Multiply) 执行加权平均
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# --- 准备训练集的键和值 (Leave-One-Out 遮蔽) ---
X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
# mask 掉自己对自己的注意力 (对角线置为0)，形状变为 (n_train, n_train-1)
mask = (1 - torch.eye(n_train)).type(torch.bool)
keys = X_tile[mask].reshape((n_train, -1))
values = Y_tile[mask].reshape((n_train, -1))

# --- 训练网络 ---
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

epochs = 5
loss_history = []
pbar = tqdm(range(epochs), desc="Training Model")  # tqdm 进度条

for epoch in pbar:
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    
    current_loss = float(l.sum())
    loss_history.append(current_loss)
    # 更新进度条显示的 loss
    pbar.set_postfix({'loss': f'{current_loss:.6f}'})

# 绘制损失下降曲线 (替代 d2l.Animator)
plt.figure(figsize=(5, 3))
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()

# --- 测试与预测 ---
# 测试阶段，每个 query (x_test) 都能看到所有的 x_train，所以形状为 (n_test, n_train)
keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))

# 推理时禁用梯度
with torch.no_grad():
    # 修复了原代码中的瑕疵: 去掉了无意义的 .unsqueeze(1)，保持一维用于 matplotlib 绘图
    y_hat = net(x_test, keys, values)  

# 绘制参数化模型的拟合图与热图
plot_kernel_reg(y_hat, title="Parametric Attention")
show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
              xlabel='Sorted training inputs',
              ylabel='Sorted testing inputs')