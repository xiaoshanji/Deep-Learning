import os
import math
import collections
import requests
import zipfile
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 通用工具与设备函数
# ==========================================

def try_gpu():
    """如果存在GPU，则返回gpu(0)，否则返回cpu()"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 数据处理与词表构建 (真实 fra-eng 数据集)
# ==========================================

def download_extract_nmt():
    """下载并解压真实的法语-英语机器翻译数据集"""
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'fra-eng.zip')
    extract_dir = os.path.join(data_dir, 'fra-eng')
    data_path = os.path.join(extract_dir, 'fra.txt')

    # 如果本地没有数据，则自动下载
    if not os.path.exists(data_path):
        print("正在下载真实的 fra-eng 数据集，请稍候...")
        r = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("下载并解压完成！")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return f.read()

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None: tokens = []
        if reserved_tokens is None: reserved_tokens = []
        counter = collections.Counter([token for line in tokens for token in line])
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq: break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self): return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    @property
    def unk(self): return 0

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = download_extract_nmt()
    
    # 将标点符号与单词分开
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    text = ''.join(out)
    
    # 词元化
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
            
    # 构建词表
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    
    # 截断或填充序列
    def truncate_pad(line, num_steps, padding_token):
        if len(line) > num_steps: return line[:num_steps]
        return line + [padding_token] * (num_steps - len(line))
        
    src_arrays, tgt_arrays, src_valid_lens, tgt_valid_lens = [], [], [], []
    for src, tgt in zip(source, target):
        src_seq = src_vocab[src] + [src_vocab['<eos>']]
        tgt_seq = tgt_vocab[tgt] + [tgt_vocab['<eos>']]
        src_valid_lens.append(len(src_seq))
        tgt_valid_lens.append(len(tgt_seq))
        src_arrays.append(truncate_pad(src_seq, num_steps, src_vocab['<pad>']))
        tgt_arrays.append(truncate_pad(tgt_seq, num_steps, tgt_vocab['<pad>']))
        
    # 转换为 Dataset
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(src_arrays), torch.tensor(src_valid_lens),
        torch.tensor(tgt_arrays), torch.tensor(tgt_valid_lens)
    )
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab

# ==========================================
# 3. 核心数学运算与掩码机制 (Masking)
# ==========================================

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项（即Padding部分），防止其参与计算"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作，用于注意力机制"""
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 将被掩码的值设为一个非常小的负数(-1e6)，这样Softmax后其概率就趋近于0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# ==========================================
# 4. 模型架构构建：编码器、注意力层与解码器
# ==========================================

class Seq2SeqEncoder(nn.Module):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # 在RNN中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # output的形状: (num_steps, batch_size, num_hiddens)
        # state的形状: (num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X)
        return output, state

class AdditiveAttention(nn.Module):
    """加性注意力机制 (Additive Attention)
    数学原理：当 Query 和 Key 的维度不一致，或者进行相似度计算时，加性注意力将两者映射到同一维度后相加。
    公式: a(q, k) = w_v * tanh(W_q * q + W_k * k)
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 扩展维度后相加。queries形状：(batch_size, 查询个数, 1, num_hiddens)
        # keys形状：(batch_size, 1, 键值对个数, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        # scores形状：(batch_size, 查询个数, 键值对个数)
        scores = self.w_v(features).squeeze(-1)
        # 过滤掉 Padding 词汇的注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # values形状：(batch_size, 键值对个数, 值的维度)
        # bmm为批量矩阵乘法，生成最终的上下文向量 (Context Vector)
        return torch.bmm(self.dropout(self.attention_weights), values)

class Decoder(nn.Module):
    """解码器基础接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    def forward(self, X, state):
        raise NotImplementedError

class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口 (用户保留代码)"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """带注意力机制的序列到序列解码器 (用户保留代码)"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # enc_outputs 包含编码器的 (outputs, hidden_state)
        outputs, hidden_state = enc_outputs
        # 将 outputs 的时间步维度调整到第二维 (batch_size, num_steps, num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输入X的形状转为(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        
        # 按时间步遍历解码器输入
        for x in X:
            # Query: 解码器上一时间步的顶层隐状态, 形状为(batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # 计算上下文向量 (Context Vector), 结合了所有编码器时间步的信息
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 将上下文向量与当前输入 x 拼接 (Concatenation)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 变形后送入RNN
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
            
        # 经过全连接层映射到词汇表大小
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

class EncoderDecoder(nn.Module):
    """编码器-解码器架构"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# ==========================================
# 5. 训练与评估工具
# ==========================================

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型 (加入 tqdm 进度条)"""
    def xavier_init_weights(m):
        """Xavier 权重初始化"""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    print("开始训练 Seq2Seq 模型...")
    # 包装 tqdm 进度条
    pbar = tqdm(range(num_epochs), desc="Training Epochs")
    for epoch in pbar:
        total_loss, total_tokens = 0.0, 0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            
            # 强制教学 (Teacher Forcing): 解码器输入始终为真实的标签序列（错位一个单位）
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            
            # 前向传播
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            
            # 梯度裁剪：防止 RNN 的梯度爆炸问题
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            
            total_loss += l.sum().item()
            total_tokens += num_tokens.item()
            
        # 在进度条上更新平均损失
        pbar.set_postfix({'Loss': f"{total_loss / total_tokens:.4f}"})

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """在序列到序列模型中预测翻译"""
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = src_tokens[:num_steps] if len(src_tokens) > num_steps else src_tokens + [src_vocab['<pad>']] * (num_steps - len(src_tokens))
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 解码器的第一个输入是 <bos>
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用最高概率的词作为下一个时间步的输入 (贪心搜索 Greedy Search)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']: break
        output_seq.append(pred)
        
    return ' '.join([tgt_vocab.idx_to_token[p] for p in output_seq]), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    """计算BLEU分数: 评估生成文本与真实文本的匹配度"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(4.5, 4)):
    """绘制注意力热力图"""
    fig, axes = plt.subplots(matrices.shape[0], matrices.shape[1], figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap='Blues')
            if i == matrices.shape[0] - 1: ax.set_xlabel(xlabel)
            if j == 0: ax.set_ylabel(ylabel)
            if titles: ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.8)
    plt.show()

# ==========================================
# 6. 主执行流程 (完全按照您的代码结构)
# ==========================================

if __name__ == "__main__":
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, try_gpu()

    # 1. 下载读取数据集，并将所有 d2l 函数转化为上述原生 PyTorch 函数
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    
    # 2. 实例化编码器和带有 Attention 的解码器
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    
    # 3. 运行含 TQDM 的训练
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 4. 预测与 BLEU 分数计算
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    
    print("\n预测结果:")
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

    # 5. 注意力权重可视化 (针对最后一个句子)
    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
        1, 1, -1, num_steps))

    # 加上一个包含序列结束词元
    show_heatmaps(
        attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
        xlabel='Key positions (源语言)', ylabel='Query positions (目标语言)')