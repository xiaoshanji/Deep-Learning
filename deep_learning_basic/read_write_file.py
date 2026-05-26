import torch
from torch import nn
from torch.nn import functional as F

# 保存和加载张量
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
# x2

# 保存和加载模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
Y_clone = clone(X)
print(Y_clone == Y)