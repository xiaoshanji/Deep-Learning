import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. 生成序列数据
# ==========================================
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# 绘制原始数据
plt.figure(figsize=(6, 3))
plt.plot(time.numpy(), x.numpy(), label='data')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([1, 1000])
plt.legend()
plt.show()

# ==========================================
# 2. 数据预处理 (马尔可夫假设构建特征与标签)
# ==========================================
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600

# 使用 PyTorch 原生 DataLoader 替换 d2l.load_array
dataset = TensorDataset(features[:n_train], labels[:n_train])
train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 3. 模型定义与训练
# ==========================================
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction='none')

# 替代 d2l.evaluate_loss 的辅助函数
def evaluate_loss(net, data_iter, loss_fn):
    """计算评估数据集上的损失"""
    net.eval()  # 设置为评估模式
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            l = loss_fn(net(X), y)
            total_loss += l.sum().item()
            total_samples += l.numel()
    return total_loss / total_samples

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        net.train()  # 设置为训练模式
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        
        # 打印每个 epoch 的平均损失
        epoch_loss = evaluate_loss(net, train_iter, loss)
        print(f'epoch {epoch + 1}, loss: {epoch_loss:f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# ==========================================
# 4. 预测与可视化
# ==========================================

# -- 单步预测 --
net.eval()
with torch.no_grad():
    onestep_preds = net(features)

plt.figure(figsize=(6, 3))
plt.plot(time.numpy(), x.numpy(), label='data')
plt.plot(time[tau:].numpy(), onestep_preds.numpy(), label='1-step preds')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([1, 1000])
plt.legend()
plt.show()


# -- 多步预测 --
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
with torch.no_grad():
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))

plt.figure(figsize=(6, 3))
plt.plot(time.numpy(), x.numpy(), label='data')
plt.plot(time[tau:].numpy(), onestep_preds.numpy(), label='1-step preds')
plt.plot(time[n_train + tau:].numpy(), multistep_preds[n_train + tau:].numpy(), label='multistep preds')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([1, 1000])
plt.legend()
plt.show()


# -- k步预测细节 --
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
with torch.no_grad():
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
plt.figure(figsize=(6, 3))
for i in steps:
    plt.plot(time[tau + i - 1: T - max_steps + i].numpy(), 
             features[:, (tau + i - 1)].numpy(), 
             label=f'{i}-step preds')
plt.xlabel('time')
plt.ylabel('x')
plt.xlim([5, 1000])
plt.legend()
plt.show()