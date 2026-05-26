import torch
from torch import nn
from fashion_mnist import load_data_fashion_mnist
import matplotlib.pyplot as plt

# 1. 检查并设置 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Training on {device}')

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    # 【修改点】添加 device=X.device，确保生成的随机 mask 和 X 在同一个设备（GPU）上
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    return mask * X / (1.0 - dropout)

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

# 实例化模型
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
# 【修改点】将模型移动到 GPU
net.to(device)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')



if __name__ == '__main__':

    # 2. 替换 d2l，使用 torchvision 和 DataLoader 加载 Fashion-MNIST
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    test_accs = []

    # 3. 替换 d2l.train_ch3，手写训练循环
    for epoch in range(num_epochs):
        net.train()  # 将模型设置为训练模式（这会自动将 self.training 设为 True）
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    
        for X, y in train_iter:
        # 【修改点】将数据移动到 GPU
            X, y = X.to(device), y.to(device)
        
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y).mean() # 取均值计算当前 batch 的损失
            l.backward()
            trainer.step()
        
            # 统计训练损失和准确率
            train_loss_sum += l.item() * y.shape[0]
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        
        # 测试集评估
        net.eval()  # 将模型设置为评估模式（这会自动将 self.training 设为 False，关闭 Dropout）
        test_acc_sum, test_n = 0.0, 0
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                test_n += y.shape[0]
        
        # 计算当前 Epoch 的各项指标
        epoch_loss = train_loss_sum / n
        epoch_train_acc = train_acc_sum / n
        epoch_test_acc = test_acc_sum / test_n
    
        # 将计算好的指标追加到列表中
        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)

        print(f'Epoch {epoch + 1}, Loss: {train_loss_sum / n:.4f}, '
          f'Train Acc: {train_acc_sum / n:.4f}, Test Acc: {test_acc_sum / test_n:.4f}')
    
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 绘制 Loss 曲线 (使用左侧 Y 轴) ---
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='tab:blue', fontsize=12)
    line1 = ax1.plot(epochs, train_losses, color='tab:blue', label='Train Loss', marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.6) # 添加网格线

    # --- 绘制 Accuracy 曲线 (使用右侧共享的 Y 轴) ---
    ax2 = ax1.twinx()  # 实例化共享相同 X 轴的第二个 Y 轴
    ax2.set_ylabel('Accuracy', color='tab:red', fontsize=12) 
    line2 = ax2.plot(epochs, train_accs, color='tab:orange', label='Train Acc', linestyle='--', marker='s', linewidth=2)
    line3 = ax2.plot(epochs, test_accs, color='tab:green', label='Test Acc', linestyle='-.', marker='^', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # --- 合并图例 ---
    # 将三个线条的 label 收集起来，统一放在一个图例框中
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('Training Process: Loss and Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
    plt.show()
