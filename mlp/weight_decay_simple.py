import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础数据处理与评估函数
# ==========================================

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) 
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    total_loss = 0.0
    total_samples = 0
    # 评估模式下不需要计算梯度，节省内存和算力
    with torch.no_grad(): 
        for X, y in data_iter:
            l = loss(net(X), y)
            total_loss += l.sum().item()
            total_samples += l.numel()
    return total_loss / total_samples

# ==========================================
# 2. 数据准备
# ==========================================

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

train_X, train_y = synthetic_data(true_w, true_b, n_train)
train_iter = load_array((train_X, train_y), batch_size)

test_X, test_y = synthetic_data(true_w, true_b, n_test)
test_iter = load_array((test_X, test_y), batch_size, is_train=False)

# ==========================================
# 3. 简洁版训练模型 (使用 PyTorch 框架)
# ==========================================

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    
    # 核心：PyTorch 通过优化器直接提供权重衰减功能（weight_decay）
    # 这里设定偏置参数(bias)没有衰减，只有权重参数(weight)有衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}], lr=lr)
    
    # 记录列表，替代 d2l 的 Animator
    train_ls, test_ls = [], []
    epochs_recorded = []

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
            
        if (epoch + 1) % 5 == 0:
            train_ls.append(evaluate_loss(net, train_iter, loss))
            test_ls.append(evaluate_loss(net, test_iter, loss))
            epochs_recorded.append(epoch + 1)
            
    print(f'wd={wd} 时，w的L2范数：', net[0].weight.norm().item())
    
    # 使用 matplotlib 绘图
    plt.figure(figsize=(5, 3))
    plt.plot(epochs_recorded, train_ls, label='train')
    plt.plot(epochs_recorded, test_ls, label='test', linestyle='--')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.title(f'Weight Decay (wd={wd})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

# 运行对比
train_concise(0) # 无权重衰减，极易过拟合
# train_concise(3) # 加入权重衰减，缓解过拟合