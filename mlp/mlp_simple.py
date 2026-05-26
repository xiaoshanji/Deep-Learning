import torch
from torch import nn
from fashion_mnist import load_data_fashion_mnist
import matplotlib.pyplot as plt

# 确保 softmax 模块中的 train_ch3 是带 device 参数的 GPU 版本
from softmax import train_ch3, predict_ch3

# ==========================================
# 1. 自动检测并定义设备 (GPU/CPU)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用的设备: {device}")

# ==========================================
# 2. 搭建模型结构
# ==========================================
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 初始化权重
net.apply(init_weights)

# 【核心新增：GPU 适配】将整个网络及其内部的所有参数一键移动到 GPU 上
net = net.to(device)

# ==========================================
# 3. 超参数、损失函数与优化器
# ==========================================
batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss(reduction='none')

# 因为 net 已经被移动到了 GPU，此时 net.parameters() 返回的也是 GPU 上的参数
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# ==========================================
# 4. 主程序执行入口
# ==========================================
if __name__ == '__main__':
    # 获取数据迭代器
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    
    print("开始训练...")
    # 【修改：GPU 适配】额外传入 device 参数
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer, device)
    
    # 确保训练折线图在 IDE 中驻留
    print("训练完成！")
    plt.show()


    print("正在进行预测...")
    # 【修改：GPU 适配】额外传入 device 参数
    predict_ch3(net, test_iter, device)