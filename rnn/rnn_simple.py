import math
import time
import collections
import re
import os
import urllib.request
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 辅助函数与数据加载模块 (替代 d2l)
# ==========================================
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def download_time_machine():
    data_dir = './data'
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    fname = os.path.join(data_dir, 'timemachine.txt')
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    return fname

def load_corpus_time_machine(max_tokens=-1):
    with open(download_time_machine(), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    text = ' '.join([re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines])
    tokens = list(text) 
    
    counter = collections.Counter(tokens)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    idx_to_token = ['<unk>']
    token_to_idx = {'<unk>': 0}
    for token, freq in token_freqs:
        if token not in token_to_idx:
            idx_to_token.append(token)
            token_to_idx[token] = len(idx_to_token) - 1
            
    corpus = [token_to_idx.get(token, 0) for token in tokens]
    if max_tokens > 0: corpus = corpus[:max_tokens]
        
    class SimpleVocab:
        def __init__(self, i2t, t2i):
            self.idx_to_token = i2t
            self.token_to_idx = t2i
        def __len__(self): return len(self.idx_to_token)
        def __getitem__(self, tokens):
            if not isinstance(tokens, (list, tuple)):
                return self.token_to_idx.get(tokens, 0)
            return [self.__getitem__(token) for token in tokens]
            
    return corpus, SimpleVocab(idx_to_token, token_to_idx)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    import random
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, max_tokens=10000):
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        return seq_data_iter_sequential(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, max_tokens)
    return data_iter, data_iter.vocab

# ==========================================
# 核心模型：使用 PyTorch nn.RNN
# ==========================================
class RNNModel(nn.Module):
    """循环神经网络模型 (基于框架实现)"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 判断是否为双向 RNN
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # 1. 转换形状为 (时间步, 批量大小) 并进行 One-Hot 编码
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        
        # 2. 将数据输入给 nn.RNN，Y 为所有时间步的隐状态集合
        Y, state = self.rnn(X, state)
        
        # 3. 将 Y 展平为 (时间步数 * 批量大小, 隐层维度)，再通过线性层
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.RNN 和 nn.GRU 返回单一张量
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), 
                                device=device)
        else:
            # nn.LSTM 返回 (隐状态, 记忆细胞) 元组
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))

# ==========================================
# 训练与预测逻辑
# ==========================================
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    for y in prefix[1:]:  
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
        
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # 取全连接层输出最大概率的索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度"""
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device):
    """训练网络一个迭代周期"""
    state, timer = None, Timer()
    metric = Accumulator(2)
    
    for X, Y in train_iter:
        if state is None:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 剥离状态计算图
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
                    
        # 确保标签 Y 展平的方式和预测值 Y_hat 展平的方式一致！
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()
            
        metric.add(l * y.numel(), y.numel())
        
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device):
    """带 tqdm 的完整训练流程"""
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr)
    
    history_ppl = []
    epochs_lst = []
    
    pbar = tqdm(range(num_epochs), desc="Training nn.RNN", unit="epoch")
    for epoch in pbar:
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device)
        
        if (epoch + 1) % 10 == 0:
            history_ppl.append(ppl)
            epochs_lst.append(epoch + 1)
            pbar.set_postfix({'PPL': f'{ppl:.2f}'})
            
    print(f'\n最终困惑度: {ppl:.1f}, {speed:.1f} 词元/秒 on {str(device)}')
    print(f"预测结果 1: {predict_ch8('time traveller', 20, net, vocab, device)}")
    print(f"预测结果 2: {predict_ch8('traveller', 20, net, vocab, device)}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_lst, history_ppl, marker='o', color='blue', label='Train PPL')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('PyTorch nn.RNN Training Progress')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

# ==========================================
# 执行入口
# ==========================================
if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    
    num_hiddens = 256
    vocab_size = len(vocab)
    
    # print("================ 查看 Vocab ================")
    # # 1. 查看词表的总大小
    # print(f"词表总大小: {len(vocab)}")

    # # 2. 查看索引到字符的映射 (idx_to_token)
    # # 它是一个列表，按照字符在文章中出现的频率从高到低排列（0固定为<unk>）
    # print(f"出现频率最高的前 10 个字符: {vocab.idx_to_token[:10]}")

    # # 3. 查看字符到索引的映射 (token_to_idx)
    # # 测试几个具体字符对应的数字
    # print(f"空格 ' ' 对应的数字是: {vocab.token_to_idx.get(' ')}")
    # print(f"字母 'e' 对应的数字是: {vocab.token_to_idx.get('e')}")
    # print(f"字母 't' 对应的数字是: {vocab.token_to_idx.get('t')}")

    # print("\n============== 查看 train_iter ==============")
    # # 手动从迭代器中抽取第一个 Batch
    # X, Y = next(iter(train_iter))

    # # 1. 查看张量的形状 (Shape)
    # # 预期输出: X 形状: torch.Size([32, 35]), Y 形状: torch.Size([32, 35])
    # print(f"输入 X 的形状 (批量大小, 时间步数): {X.shape}")
    # print(f"标签 Y 的形状 (批量大小, 时间步数): {Y.shape}")

    # # 2. 查看张量里面具体的纯数字
    # # 打印第一个批次中，第一句话（第0行）的前 10 个数字
    # print(f"\nX 第一行的前 10 个数字:\n{X[0, :10]}")
    # print(f"Y 第一行的前 10 个数字:\n{Y[0, :10]}")



    # print("\n============= 数据还原与错位验证 =============")
    # # 提取第一句话所有的数字，并用 vocab 翻译回字符
    # text_X = ''.join([vocab.idx_to_token[idx] for idx in X[0]])
    # text_Y = ''.join([vocab.idx_to_token[idx] for idx in Y[0]])

    # print(f"模型的输入 X (过去的字符):\n'{text_X}'\n")
    # print(f"模型的标签 Y (未来的字符):\n'{text_Y}'")

    # 实例化官方的 RNN 层
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
    device = try_gpu()
    
    # 包装进我们的自定义网络
    net = RNNModel(rnn_layer, vocab_size=vocab_size).to(device)
    
    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)