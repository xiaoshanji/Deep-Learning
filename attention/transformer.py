import os
import math
import collections
import urllib.request
import zipfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据集下载与预处理 (替代 d2l.load_data_nmt)
# ==========================================

def download_extract_nmt():
    """下载并解压英法翻译数据集"""
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'fra-eng.zip')
    
    if not os.path.exists(filepath):
        print("正在下载英法机器翻译数据集...")
        urllib.request.urlretrieve(url, filepath)
        
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    with open(os.path.join(data_dir, 'fra-eng', 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    # 使用空格替换不间断空格，转小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

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

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps: return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):
    """将文本序列转换成小批量序列"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(download_extract_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    dataset = TensorDataset(src_array, src_valid_len, tgt_array)
    data_iter = DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab

# ==========================================
# 2. Transformer 核心组件 (已剔除 d2l 依赖)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(ffn_num_input, ffn_num_hiddens),
            nn.ReLU(),
            nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        )

    def forward(self, X):
        return self.net(X)

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout=dropout, batch_first=True)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, key_padding_mask):
        attn_output, attn_weights = self.attention(X, X, X, key_padding_mask=key_padding_mask, average_attn_weights=False)
        self.attention_weights = attn_weights 
        Y = self.addnorm1(X, attn_output)
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.ModuleList([
            EncoderBlock(num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, valid_lens=None):
        key_padding_mask = None
        if valid_lens is not None:
            max_len = X.size(1)
            key_padding_mask = torch.arange(max_len, device=X.device)[None, :] >= valid_lens[:, None]

        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for blk in self.blks:
            X = blk(X, key_padding_mask)
        return X, key_padding_mask

class DecoderBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.attention1 = nn.MultiheadAttention(num_hiddens, num_heads, dropout=dropout, batch_first=True)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = nn.MultiheadAttention(num_hiddens, num_heads, dropout=dropout, batch_first=True)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, enc_outputs, enc_padding_mask, tgt_mask):
        attn1_out, self.attn_weights1 = self.attention1(X, X, X, attn_mask=tgt_mask, average_attn_weights=False)
        Y = self.addnorm1(X, attn1_out)
        
        attn2_out, self.attn_weights2 = self.attention2(Y, enc_outputs, enc_outputs, key_padding_mask=enc_padding_mask, average_attn_weights=False)
        Z = self.addnorm2(Y, attn2_out)
        return self.addnorm3(Z, self.ffn(Z))

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.ModuleList([
            DecoderBlock(num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, enc_outputs, enc_padding_mask):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        seq_len = X.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=X.device)

        for blk in self.blks:
            X = blk(X, enc_outputs, enc_padding_mask, tgt_mask)
        return self.dense(X)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs, enc_padding_mask = self.encoder(enc_X, enc_valid_lens)
        return self.decoder(dec_X, enc_outputs, enc_padding_mask)

# ==========================================
# 3. 训练、预测与评估函数
# ==========================================

def train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        pbar = tqdm(train_iter, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for X, X_valid_lens, Y in pbar:
            optimizer.zero_grad()
            X, X_valid_lens, Y = X.to(device), X_valid_lens.to(device), Y.to(device)
            
            # 强制教学: 解码器输入 Y 的前 n-1 个元素，加上 <bos> 在数据集构建时已经隐含
            # (通常Y[:,0]是首词，在此简略实现中直接利用偏移)
            bos_tensor = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos_tensor, Y[:, :-1]], 1)
            
            Y_hat = net(X, dec_input, X_valid_lens)
            loss = criterion(Y_hat.reshape(-1, Y_hat.shape[-1]), Y.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split()] + [src_vocab['<eos>']]
    enc_valid_lens = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)

    with torch.no_grad():
        enc_outputs, enc_padding_mask = net.encoder(enc_X, enc_valid_lens)
        dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], device=device), dim=0)
        output_seq = []
        
        for _ in range(num_steps):
            Y_hat = net.decoder(dec_X, enc_outputs, enc_padding_mask)
            pred = Y_hat.argmax(dim=-1)[:, -1]
            if pred.item() == tgt_vocab['<eos>']:
                break
            output_seq.append(pred.item())
            dec_X = torch.cat([dec_X, pred.unsqueeze(0)], dim=1)
            
    return ' '.join(tgt_vocab.to_tokens(output_seq))

def bleu(pred_seq, label_seq, k=2):
    pred_tokens, label_tokens = pred_seq.split(), label_seq.split()
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    if len_pred == 0 or len_label == 0: return 0.0
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        label_subs = collections.Counter([' '.join(label_tokens[i:i+n]) for i in range(len_label - n + 1)])
        pred_subs = collections.Counter([' '.join(pred_tokens[i:i+n]) for i in range(len_pred - n + 1)])
        for sub in pred_subs:
            num_matches += min(pred_subs[sub], label_subs[sub])
        score *= math.pow(num_matches / (len_pred - n + 1) if len_pred - n + 1 > 0 else 0, math.pow(0.5, n))
    return score

def show_heatmaps(matrices, xlabel, ylabel, titles, figsize=(7, 3.5)):
    plt.figure(figsize=figsize)
    for i, matrix in enumerate(matrices):
        plt.subplot(1, len(matrices), i + 1)
        sns.heatmap(matrix.detach().numpy(), cmap="viridis", cbar=(i == len(matrices)-1))
        plt.title(titles[i])
        plt.xlabel(xlabel)
        if i == 0:
            plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 执行训练与测试
# ==========================================

# 超参数
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
norm_shape = [32]

# 加载原书使用的前600条英法翻译数据集 (为了快速跑通，实际可调大 num_examples)
print("准备数据集...")
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, num_examples=600)

encoder = TransformerEncoder(
    len(src_vocab), num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)

print(f"在 {device} 上开始训练...")
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# 测试和评估
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

print("\n翻译测试:")
for eng, fra in zip(engs, fras):
    translation = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu: {bleu(translation, fra, k=2):.3f}')

# 绘制最后一层Encoder的注意力热力图
print("\n绘制注意力热力图...")
sample_attn_weights = net.encoder.blks[-1].attention_weights[0] 
show_heatmaps(
    [sample_attn_weights[i].cpu() for i in range(num_heads)], 
    xlabel='Key positions',
    ylabel='Query positions',
    titles=[f'Head {i+1}' for i in range(num_heads)]
)