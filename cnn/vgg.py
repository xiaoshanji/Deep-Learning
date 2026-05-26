import time
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 替代 d2l 的辅助函数与类
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
    """用GPU训练模型，并在结束后绘制指标趋势图"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer = Timer()
    num_batches = len(train_iter)
    
    history_train_l, history_train_acc, history_test_acc = [], [], []
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
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
            
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        history_train_l.append(train_l)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        
        print(f'epoch {epoch + 1}, loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
        
    print(f'Final: loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    
    # 绘制折线图
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_train_l, label='train loss', color='blue', linestyle='dashed', marker='o')
    plt.plot(epochs, history_train_acc, label='train acc', color='red', marker='s')
    plt.plot(epochs, history_test_acc, label='test acc', color='green', marker='^')
    
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('VGG-11 Training Metrics on Fashion-MNIST')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


# ==========================================
# VGG-11 模型定义
# ==========================================

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


if __name__ == '__main__':
    
    # 经典 VGG-11 架构：(1+1+2+2+2=8个卷积层)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    net = vgg(conv_arch)

    print("===== 网络结构测试 =====")
    X = torch.randn(size=(1, 1, 224, 224))
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)
    print("========================\n")

    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    
    # 启动训练和可视化
    train_ch6_with_plot(net, train_iter, test_iter, num_epochs, lr, try_gpu())