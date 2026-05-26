import time
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度条

# ==========================================
# 替代 d2l 的底层基础组件
# ==========================================

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    
    return (DataLoader(mnist_train, batch_size, shuffle=True, num_workers=2),
            DataLoader(mnist_test, batch_size, shuffle=False, num_workers=2))

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def sum(self):
        return sum(self.times)

def evaluate_accuracy_gpu(net, data_iter, device=None): 
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# ==========================================
# 训练与可视化模块
# ==========================================

def train_ch6_with_plot(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型，使用 tqdm 显示进度，结束后绘制指标趋势图"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print(f'Training on {device}...\n')
    net.to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer = Timer()
    
    history_train_l, history_train_acc, history_test_acc = [], [], []
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        
        # tqdm 进度条包装
        pbar = tqdm(train_iter, desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch')
        
        for X, y in pbar:
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            
            current_loss = metric[0] / metric[2]
            current_acc = metric[1] / metric[2]
            pbar.set_postfix({'loss': f'{current_loss:.3f}', 'acc': f'{current_acc:.3f}'})
            
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        history_train_l.append(train_l)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        
        print(f'Epoch {epoch + 1} Summary -> test acc: {test_acc:.3f}\n')
        
    print(f'=== Final Results ===')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'Throughput: {metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    
    # 绘制训练曲线
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_train_l, label='train loss', color='blue', linestyle='dashed', marker='o')
    plt.plot(epochs, history_train_acc, label='train acc', color='red', marker='s')
    plt.plot(epochs, history_test_acc, label='test acc', color='green', marker='^')
    
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('DenseNet Training Metrics')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


# ==========================================
# DenseNet 模型定义
# ==========================================

def conv_block(input_channels, num_channels):
    """标准的 BN -> ReLU -> Conv 组合（前向激活）"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    """稠密块：将所有层的输出拼接在一起"""
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # 每多一个卷积层，输入通道数就多出之前所有层的输出
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 核心操作：在通道维度上拼接输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    """过渡层：用于控制通道数不要爆炸，并缩小特征图尺寸"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

# --- 网络组装 ---
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# num_channels: 当前通道数初始为64
# growth_rate: 增长率，也是DenseBlock中每个卷积层的输出通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 更新当前通道数：当前通道数 + (卷积层数量 * 每次增长的通道数)
    num_channels += num_convs * growth_rate
    
    # 在稠密块之间添加一个过渡层，使通道数量直接减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))


if __name__ == '__main__':
    
    print("===== DenseNet 网络层级推导 =====")
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    print("=================================\n")

    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    
    train_ch6_with_plot(net, train_iter, test_iter, num_epochs, lr, try_gpu())