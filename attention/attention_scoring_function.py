import math
import torch
from torch import nn

def sequence_mask(X, valid_lens, value=0):
    """在序列中掩蔽（Mask）不相关的项（即填充的 <pad> token）"""
    maxlen = X.size(1)
    # 创建一个形状为 (1, maxlen) 的索引序列，并与有效长度比较，生成布尔掩码
    mask = torch.arange((maxlen), dtype=torch.float32, 
                        device=X.device)[None, :] < valid_lens[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行 softmax 操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 将张量展平为 2D 以便使用 sequence_mask
        # 最后一轴上被掩蔽的元素使用一个非常大的负值 (-1e6) 替换
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """加性注意力 (Additive Attention)"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size, 查询的个数, 1, num_hiddens)
        # keys的形状：(batch_size, 1, “键－值”对的个数, num_hiddens)
        # 此时可以使用广播机制 (Broadcasting) 进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        # self.w_v 仅有一个输出 (映射到1维)，因此将其从形状中 squeeze (移除)
        # scores的形状：(batch_size, 查询的个数, “键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # values的形状：(batch_size, “键－值”对的个数, 值的维度)
        # 使用 bmm (批量矩阵乘法) 计算加权和
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    """缩放点积注意力 (Scaled Dot-Product Attention)"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # queries的形状：(batch_size, 查询的个数, d)
        # keys的形状：(batch_size, “键－值”对的个数, d)
        # transpose(1,2) 交换keys的序列长度维度和特征维度，使其可以进行矩阵乘法
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)