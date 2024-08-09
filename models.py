import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, device):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, d_model, device = device)
        position = torch.arange(0, seq_len, dtype = torch.float, device = device).unsqueeze(1)
        _2i = torch.arange(0, d_model, step = 2, dtype = torch.float, device = device)
        pe[:, ::2] = torch.sin(position/10000**(_2i/d_model))
        pe[:, 1::2] = torch.cos(position/10000**(_2i/d_model))
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model # 총 차원
        self.n_heads = n_heads # head 수
        self.head_dim = d_model // n_heads # head 차원

        self.W_q = nn.Linear(d_model, d_model, bias = False) # batch_size x n x d_q
        self.W_k = nn.Linear(d_model, d_model, bias = False) # batch_size x n x d_k
        self.W_v = nn.Linear(d_model, d_model, bias = False) # batch_size x n x d_v

        self.W_o = nn.Linear(d_model, d_model, bias = False) # concat 후 linear, 각 head가 따로 학습한 정보를 취합하는 용도

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def split_heads(self, x): # multi-head attention을 위해 여러 head를 만듦
        batch_size, seq_length, d_model = x.size() # [배치 크기, 문장 길이, 각 단어의 차원]
        # d_model(전체 차원)을 지정한 head의 수로 나눔
        return x.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2) 
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q: [batch_size, n_head, seq_len, dimension]
        # k.transpose(-2, -1): [batch_size, n_head, dimension, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # attn_scores: [batch_size, n_head, seq_len, seq_len]
        if mask is not None: # masked_multi_head_attention
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # 음수 무한대로 줘서 softmax 취할시 0으로 변하게 함
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        # attn_probs: [batch_size, n_head, seq_len, seq_len]
        # v: [batch_size, n_head, seq_len, dimension]
        output = torch.matmul(attn_probs, v) # 각 단어의 attention 값을 담은 value
        # output: [batch_size, n_head, seq_len, dimension]
        return output, attn_probs
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 
    
    def forward(self, query, key, value, mask=None):
        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))
        # query와 value를 내적한 값의 scaled된 attention 값
        attn_output, attention = self.scaled_dot_product_attention(query, key, value, mask)
        # attn_output: [batch_size, n_head, seq_len, dimension]
        # self.combine_heads(attn_output) -> [batch_size, seq_len, n_head x dimension]
        output = self.W_o(self.combine_heads(attn_output))
        # output: [batch_size, seq_len, d_model]
        return output, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, pff_dim, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.fc_1 = nn.Linear(d_model, pff_dim)
        self.fc_2 = nn.Linear(pff_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model], applied to each position separately and identically
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x: [batch_size, seq_len, pff_dim]
        x = self.fc_2(x)
        # x: [batch_size, seq_len, d_model]
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout, device)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [batch_size, seq_len, d_model]
        attn_output, _ = self.self_attn(x, x, x, mask)
        # attn_output: [batch_size, seq_len, d_model]
        x = self.norm1(x + self.dropout(attn_output)) # Add & Norm
        # x: [batch_size, seq_len, d_model]
        ff_output = self.feed_forward(x)
        # ff_output: [batch_size, seq_len, d_model]
        x = self.norm2(x + self.dropout(ff_output)) # Add & Norm
        # x: [batch_size, seq_len, d_model]
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout, device)
        self.cross_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout, device)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, trg_mask):
        # x: [batch_size, seq_len, d_model]
        attn_output, _ = self.self_attn(x, x, x, trg_mask)
        # attn_output: [batch_size, seq_len, d_model]
        x = self.norm1(x + self.dropout(attn_output)) # Add & Norm
        # x: [batch_size, seq_len, d_model]
        attn_output, attention = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output)) # Add & Norm
        # x: [batch_size, seq_len, d_model]
        ff_output = self.feed_forward(x)
        # ff_output: [batch_size, seq_len, d_model]
        x = self.norm3(x + self.dropout(ff_output)) # Add & Norm
        # x: [batch_size, seq_len, d_model]
        return x, attention

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, device)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, device) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, device) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
        self.device = device

    # def generate_mask(self, src, tar):
    #     # src_mask: [batch_size, 1, 1, seq_len]
    #     src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    #     # trg_mask: [batch_size, 1, seq_len, 1]
    #     trg_mask = (tar != 0).unsqueeze(1).unsqueeze(3).to(self.device)
    #     seq_length = tar.size(1)
    #     # nopeak_mask: [1, seq_len, seq_len]
    #     nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
    #     trg_mask = trg_mask & nopeak_mask
    #     # trg_mask: [batch_size, 1, seq_len, seq_len]
    #     return src_mask, trg_mask
    
    def make_src_mask(self, src):
        # src: [batch_size, seq_len]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # src_mask: [batch_size, 1, 1, seq_len]
        return src_mask
    
    def make_tar_mask(self, tar):
        # tar = [batch_size, tar_len]
        tar_mask = (tar != 0).unsqueeze(1).unsqueeze(3).to(self.device)
        # tar_mask: [batch_size, 1, seq_len, 1]
        seq_length = tar.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
        # nopeak_mask: [1, seq_len, seq_len]
        tar_mask = tar_mask & nopeak_mask
        # tar_mask: [batch_size, 1, seq_len, seq_len]
        return tar_mask
    
    def forward(self, src, tar):
        src_mask = self.make_src_mask(src)
        tar_mask = self.make_tar_mask(tar)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tar_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tar)))
        # src_embedded: [batch_size, seq_len, d_model]
        # tar_embedded: [batch_size, seq_len, d_model]
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        # enc_ouput: [batch_size, seq_len, d_model]
        dec_output = tar_embedded
        for dec_layer in self.decoder_layers:
            dec_output, attention = dec_layer(dec_output, enc_output, src_mask, tar_mask)
        # dec_output: [batch_size, seq_len, d_model]
        # attention: [batch_size, n_heads, trg_len, src_len]
        output = self.fc(dec_output)
        # output_shape: [batch_size, seq_len, vocab_size]
        return output, attention