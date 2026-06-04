import random
import torch
import collections
import re
import os
import urllib.request

# ==========================================
# 1. 文本预处理与加载 (替代 d2l 的底层函数)
# ==========================================

def download_time_machine():
    """下载《时间机器》数据集到本地"""
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    fname = os.path.join(data_dir, 'timemachine.txt')
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    return fname

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表 (精简版)"""
    with open(download_time_machine(), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 过滤非字母字符并转小写
    text = ' '.join([re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines])
    tokens = list(text)  # 字符级词元化
    
    # 构建词表
    counter = collections.Counter(tokens)
    token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    idx_to_token = ['<unk>']
    token_to_idx = {'<unk>': 0}
    for token, freq in token_freqs:
        if token not in token_to_idx:
            idx_to_token.append(token)
            token_to_idx[token] = len(idx_to_token) - 1
            
    # 将文本转换为索引
    corpus = [token_to_idx.get(token, 0) for token in tokens]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
        
    # 为了保持接口兼容，返回一个简单的带有 __len__ 和 to_tokens 的假 Vocab 对象
    class SimpleVocab:
        def __init__(self, i2t, t2i):
            self.idx_to_token = i2t
            self.token_to_idx = t2i
        def __len__(self):
            return len(self.idx_to_token)
        def to_tokens(self, indices):
            return [self.idx_to_token[i] for i in indices]
            
    return corpus, SimpleVocab(idx_to_token, token_to_idx)

# ==========================================
# 2. 核心：序列数据迭代器
# ==========================================

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签 (Y是X向后平移一位)
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 随机打乱起始索引
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 取出 batch_size 个随机打乱后的起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    
    # 这里的 reshape 是顺序分区的灵魂：
    # 它将连续的序列切分成 batch_size 行，保证同一行的下一个 batch 是严格连贯的。
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

# ==========================================
# 3. 数据加载器封装
# ==========================================

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
            
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


if __name__ == '__main__':
    # 为了直观演示，我们造一个简单的 0-34 的连续整数序列作为 corpus
    my_seq = list(range(35))
    batch_size = 2
    num_steps = 5

    print("===== 1. 随机抽样 (Random Iteration) =====")
    for X, Y in seq_data_iter_random(my_seq, batch_size=batch_size, num_steps=num_steps):
        print('X: ', X)
        print('Y: ', Y)
        print('-' * 20)

    print("\n===== 2. 顺序分区 (Sequential Iteration) =====")
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=batch_size, num_steps=num_steps):
        print('X: ', X)
        print('Y: ', Y)
        print('-' * 20)