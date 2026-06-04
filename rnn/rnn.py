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
# 1. 替代 d2l 的基础辅助与数据集模块
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

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# --- 数据集加载部分 (整合了之前的代码) ---
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
    """顺序分区生成迭代器"""
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
# 2. 从零实现的 RNN 核心模块
# ==========================================

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 将输入序列的批量维度和时间步维度转置，并转为 One-Hot 向量
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    

# ==========================================
# 3. 预测、梯度裁剪与训练逻辑
# ==========================================

def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 预热期：用已知的 prefix 更新隐藏状态，但不产生实际输出记录
    for y in prefix[1:]:  
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
        
    # 预测期：利用更新好的状态，不断拿自己的输出当做下一步的输入
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """裁剪梯度，防止梯度爆炸"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期"""
    state, timer = None, Timer()
    metric = Accumulator(2)  # [训练损失之和, 词元数量]
    
    for X, Y in train_iter:
        # 如果是序列开头，或者使用了随机抽样，必须初始化状态
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 顺序分区时，保留状态，但剥离计算图（防止反向传播跨越太长导致爆显存）
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
                    
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 自定义优化器，这里传入 batch_size=1 是因为上面损失已经用了 mean()
            updater(batch_size=1)
            
        metric.add(l * y.numel(), y.numel())
        
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """带进度条与画图的完整训练流程"""
    loss = nn.CrossEntropyLoss()
    
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
        
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # 用于记录画图数据
    history_ppl = []
    epochs_lst = []
    
    pbar = tqdm(range(num_epochs), desc="Training RNN", unit="epoch")
    for epoch in pbar:
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        
        # 每过10轮收集一次数据点并展示一次预测结果
        if (epoch + 1) % 10 == 0:
            history_ppl.append(ppl)
            epochs_lst.append(epoch + 1)
            pbar.set_postfix({'Perplexity': f'{ppl:.2f}'})
            
    print(f'\n最终困惑度 (Perplexity): {ppl:.1f}, {speed:.1f} 词元/秒 on {str(device)}')
    print(f"预测结果 1: {predict('time traveller ')}")
    print(f"预测结果 2: {predict('traveller ')}")
    
    # 绘制困惑度曲线
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_lst, history_ppl, marker='o', color='blue', label='Train Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity (Lower is better)')
    plt.title('RNN Training Progress')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


# ==========================================
# 4. 实例化与启动执行
# ==========================================

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    
    vocab_size = len(vocab)
    num_hiddens = 512  # 修复了原始代码缺失的隐层大小定义
    num_epochs, lr = 500, 1
    device = try_gpu()
    
    print(f"Vocab size: {vocab_size}, Hidden size: {num_hiddens}")
    
    # 实例化自建的 RNN 模型
    net = RNNModelScratch(vocab_size, num_hiddens, device, get_params,
                          init_rnn_state, rnn)
    
    # 启动顺序分区的训练
    train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False)