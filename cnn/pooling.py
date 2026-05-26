import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y



X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# pool2d(X, (2, 2))

# pool2d(X, (2, 2), 'avg')

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
# X

pool2d = nn.MaxPool2d(3)
# pool2d(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# pool2d(X)

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
# pool2d(X)

X = torch.cat((X, X + 1), 1)
# X

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
# pool2d(X) # 池化层在每个输入通道上单独运算，即输出通道数与输入通道数相同

