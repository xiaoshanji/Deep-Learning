import torch
from torch.utils import data
import matplotlib.pyplot as plt

# ==========================================
# 1. 替代 d2l 的底层函数实现
# ==========================================

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 (替代 d2l.synthetic_data)"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # 加入标准差为0.01的噪声
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器 (替代 d2l.load_array)"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def linreg(X, w, b):
    """线性回归模型 (替代 d2l.linreg)"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失 (替代 d2l.squared_loss)"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降 (替代 d2l.sgd)"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() # 梯度清零

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失 (替代 d2l.evaluate_loss)"""
    total_loss = 0.0
    total_samples = 0
    for X, y in data_iter:
        l = loss(net(X), y)
        total_loss += l.sum().item()
        total_samples += l.numel()
    return total_loss / total_samples

# ==========================================
# 2. 原始主干代码
# ==========================================

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

# 准备数据
train_X, train_y = synthetic_data(true_w, true_b, n_train)
train_iter = load_array((train_X, train_y), batch_size)

test_X, test_y = synthetic_data(true_w, true_b, n_test)
test_iter = load_array((test_X, test_y), batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    
    # 记录损失用于后续绘图
    train_ls, test_ls = [], []
    epochs_recorded = []

    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
            
        if (epoch + 1) % 5 == 0:
            train_ls.append(evaluate_loss(net, train_iter, loss))
            test_ls.append(evaluate_loss(net, test_iter, loss))
            epochs_recorded.append(epoch + 1)
            
    print('w的L2范数是：', torch.norm(w).item())
    
    # 使用 matplotlib 绘图
    plt.figure(figsize=(5, 3))
    plt.plot(epochs_recorded, train_ls, label='train')
    plt.plot(epochs_recorded, test_ls, label='test', linestyle='--')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log') # 使用对数坐标系
    plt.title(f'Weight Decay (lambd={lambd})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

# 运行对比
# train(lambd=0)
train(lambd=3)