import torch
from torch.utils import data
from torch import nn

'''用框架实现线性回归'''

# 1. 用纯 PyTorch 实现原本的 d2l.synthetic_data 函数
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    # 生成均值为0，标准差为1的随机特征 X
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算实际的 y
    y = torch.matmul(X, w) + b
    # 加入均值为0，标准差为0.01的高斯噪声
    y += torch.normal(0, 0.01, y.shape)
    # 将标签转换为列向量形式
    return X, y.reshape((-1, 1))

# 定义真实的权重和偏差
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 使用我们自己实现的函数生成数据
features, labels = synthetic_data(true_w, true_b, 1000)

# 2. 数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 3. 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)




# 定义损失函数 (均方误差)
loss = nn.MSELoss()

# 定义优化器 (随机梯度下降)
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 4. 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    # --- 训练阶段 ---
    net.train() # 可选：显式告诉模型现在是训练模式
    for X, y in data_iter:
        l = loss(net(X), y)
        l.backward()        # 反向传播计算梯度
        trainer.step()      # 更新参数
        trainer.zero_grad() # 梯度清零
    
    # --- 验证/评估阶段 ---
    net.eval() # 可选：显式告诉模型现在是评估模式 (关闭 Dropout 等)
    with torch.no_grad(): # 核心：关闭梯度计算，节省显存和算力！
        # 计算当前 epoch 的整体损失
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

# 5. 验证结果
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)