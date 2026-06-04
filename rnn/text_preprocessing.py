import collections
import re
import os
import urllib.request

# ==========================================
# 1. 数据下载与读取
# ==========================================

def download_time_machine():
    """下载《时间机器》数据集到本地 data 目录下"""
    # 指定数据存储目录
    data_dir = './data'
    # 如果目录不存在，则自动创建
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    # 拼接完整的文件路径：./data/timemachine.txt
    fname = os.path.join(data_dir, 'timemachine.txt')
    
    if not os.path.exists(fname):
        print(f'正在从 {url} 下载到 {fname}...')
        urllib.request.urlretrieve(url, fname)
    return fname

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    fname = download_time_machine()
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 将非字母字符替换为空格，去除两端空白，并统一转为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# ==========================================
# 2. 词元化与词频统计
# ==========================================

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens如果是2D列表，先将其展平成1D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# ==========================================
# 3. 词表类定义
# ==========================================

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        # 统计词频并按出现频率从高到低排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 如果输入不是列表或元组，直接返回其索引，找不到则返回<unk>的索引(0)
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        # 如果输入是列表，递归获取所有元素的索引
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  
        """未知词元的索引为0"""
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

# ==========================================
# 4. 高级封装接口
# ==========================================

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    # 默认使用字符级别的词元化
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    
    # 因为时光机器数据集中的每个文本行不一定是一个完整的句子或段落，
    # 所以将所有文本行展平到一个一维的序列列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    # 测试代码：加载数据集并打印前部分信息
    corpus, vocab = load_corpus_time_machine()
    
    print(f"词表大小: {len(vocab)}")
    print(f"语料库总词元数(字符数): {len(corpus)}")
    
    # 打印前 20 个字符及其对应的索引
    print("\n前 20 个字符及其索引映射:")
    sample_corpus = corpus[:20]
    sample_tokens = vocab.to_tokens(sample_corpus)
    for token, idx in zip(sample_tokens, sample_corpus):
        # 将空格特别显示出来以便于观察
        display_token = '[空格]' if token == ' ' else token
        print(f"'{display_token}': {idx}", end=" | ")
    print()