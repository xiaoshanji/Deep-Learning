import torch
from IPython import display
from fashion_mnist import load_data_fashion_mnist, get_fashion_mnist_labels, show_images
import matplotlib.pyplot as plt

'''softmax回归的从零实现'''

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# ==========================================
# 2. 全局超参数与模型参数
# ==========================================

num_inputs = 784
num_outputs = 10
batch_size = 256
lr = 0.1
num_epochs = 10

# 【新增: GPU 适配】自动检测并定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用的设备: {device}")

# 【修改: GPU 适配】直接在选定的设备（GPU）上初始化权重张量
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True, device=device)
b = torch.zeros(num_outputs, requires_grad=True, device=device)

# ==========================================
# 3. 核心算法与训练函数
# ==========================================

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 【修改: GPU 适配】传入 device 参数
def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # 将测试数据拉到显存
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 【修改: GPU 适配】传入 device 参数
def train_epoch_ch3(net, train_iter, loss, updater, device):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        # 将训练数据拉到显存
        X, y = X.to(device), y.to(device)
        
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def updater(batch_size):
    return sgd([W, b], lr, batch_size)

# 【修改: GPU 适配】透传 device 参数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, device):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, device)
        test_acc = evaluate_accuracy(net, test_iter, device)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 【修改: GPU 适配】传入 device，并在绘图前将数据拉回 CPU
def predict_ch3(net, test_iter, device, n=6):
    for X, y in test_iter:
        # 将数据拉到显存供模型推理
        X, y = X.to(device), y.to(device)
        break
        
    # .cpu() 将数据拉回内存，因为原生 Python 代码(如文本映射)不支持 GPU Tensor
    trues = get_fashion_mnist_labels(y.cpu())
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1).cpu())
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    
    # 绘图库 matplotlib 只能处理 CPU 上的张量，所以用 X.cpu()
    show_images(
        X[0:n].cpu().reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show() 

# ==========================================
# 4. 主程序执行入口
# ==========================================

if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    print("开始训练...")
    # 传入 device
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater, device)
    print("训练完成！\n请关闭训练损失曲线窗口以查看最终的预测结果图。")
    
    plt.show() 

    print("正在进行预测...")
    # 传入 device
    predict_ch3(net, test_iter, device)