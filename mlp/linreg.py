# 导入 random 模块，用于数据随机打乱
import random
# 导入 PyTorch 深度学习框架
import torch

'''线性回归的从零实现'''

def synthetic_data(w, b, num_examples):  #@save
    """生成 y = Xw + b + 噪声 的合成数据集"""
    # 生成服从标准正态分布的特征矩阵 X，形状为 (num_examples, len(w))
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算线性组合 Xw + b，得到准确的输出值（不含噪声）
    y = torch.matmul(X, w) + b
    # 添加均值为 0、标准差为 0.01 的高斯噪声，模拟真实观测误差
    y += torch.normal(0, 0.01, y.shape)
    # 返回特征矩阵 X 和标签列向量 y（重塑为 (num_examples, 1)）
    return X, y.reshape((-1, 1))

# 定义真实的权重和偏置，用于生成数据
true_w = torch.tensor([2, -3.4])          # 两个特征的真实权重
true_b = 4.2                              # 真实偏置
# 调用合成数据函数，生成 1000 个样本的特征和标签
features, labels = synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):
    """按批次迭代数据集，每次返回一个小批量样本"""
    # 样本总数
    num_examples = len(features)
    # 生成从 0 到 num_examples-1 的索引列表
    indices = list(range(num_examples))
    # 随机打乱索引，确保每个 epoch 中样本顺序不同（随机采样）
    random.shuffle(indices)
    # 按 batch_size 步长遍历索引列表
    for i in range(0, num_examples, batch_size):
        # 取出当前批次的索引，注意处理最后一个批次可能不足 batch_size 的情况
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        # 使用 yield 返回一个生成器，每次提供一个批次的特征和标签
        yield features[batch_indices], labels[batch_indices]

# 初始化模型参数：权重 w 形状为 (2,1)，从均值为 0、标准差 0.01 的正态分布采样，并启用梯度追踪
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# 偏置 b 初始化为标量 0，同样要求梯度
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  #@save
    """线性回归模型的前向计算"""
    # 返回预测值 y_hat = Xw + b
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失（除以 2 是为了求导后系数为 1）"""
    # 计算 (预测值 - 真实值)^2 / 2，注意将 y 的形状调整为与 y_hat 一致
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降更新参数"""
    # 在不记录梯度的上下文中执行更新操作，避免不必要的计算图构建
    with torch.no_grad():
        for param in params:
            # 参数更新公式：param = param - lr * (grad / batch_size)
            param -= lr * param.grad / batch_size
            # 将梯度清零，为下一轮计算做准备
            param.grad.zero_()

# 设置超参数
lr = 0.03          # 学习率
num_epochs = 10     # 迭代轮数（遍历整个数据集的次数）
batch_size = 10    # 每个小批量的样本数
net = linreg       # 模型函数
loss = squared_loss # 损失函数

# 训练循环
for epoch in range(num_epochs):
    # 使用 data_iter 按批次遍历数据集
    for X, y in data_iter(batch_size, features, labels):
        # 前向传播：计算当前批次预测值，并计算损失（形状为 batch_size x 1）
        l = loss(net(X, w, b), y)
        # 损失求和并反向传播，计算参数梯度
        # 由于 l 是向量，sum() 后得到一个标量，才能调用 backward()
        l.sum().backward()
        # 使用小批量随机梯度下降更新参数
        sgd([w, b], lr, batch_size)
    # 在每个 epoch 结束后，计算整个训练集上的损失（不计算梯度）
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 输出估计参数与真实参数的误差
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')