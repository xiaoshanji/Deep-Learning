import torch
from torch import nn
from fashion_mnist import get_fashion_mnist_labels, load_data_fashion_mnist
from softmax import train_ch3, predict_ch3
import matplotlib.pyplot as plt

batch_size = 256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))



def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)




if __name__ == '__main__':

    num_epochs, lr = 10, 0.1
    params = [W1, b1, W2, b2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用的设备: {device}")

    # 获取数据迭代器
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # PyTorch 内置的交叉熵损失
    loss = nn.CrossEntropyLoss(reduction='none')
    # PyTorch 内置的随机梯度下降优化器
    updater = torch.optim.SGD(params, lr=lr)

    print("开始训练...")
    # 【修改：GPU 适配】额外传入 device 参数
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, device)
    
    # 确保训练折线图不会一闪而过
    print("训练完成！\n请关闭图表窗口以查看最终的预测结果。")
    plt.show()

    print("正在进行预测...")
    # 【修改：GPU 适配】额外传入 device 参数
    predict_ch3(net, test_iter, device)