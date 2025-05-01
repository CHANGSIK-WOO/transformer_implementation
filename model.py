import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from torch import Tensor
from typing import Tuple

class InputEmbeddings(nn.Module):
    def __init__(self, d_model : int, vocab_size : int) -> None: #vocab size = 총 Token 개수
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # token index tensor x is already a shape of (batch_size, seq_len)
        # x : (batch_size, actual_seq_len, d_model)

class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model : int, seq_len : int, dropout : float) -> None : # seq_len : sentence total length
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (d_model, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # 역수는 부동 소수점 발생으로 안정성이 떨어져 지수함수로 계산 (훨씬 안정적임)

        # Apply the sine to even and the cosine to odd positions (seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0) # Add Batch Term in pe : (seq_len, d_model) --> (batch size : 1, seq_len, d_model)
        self.register_buffer('pe', pe)
        # 이건 PyTorch에서 학습하지 않는 텐서를 모듈에 등록할 때 쓰는 함수
        # pe는 위치 인코딩으로, 고정된 값이지 learnable parameter가 아님
        # 하지만 모델과 함께 .to(device)나 .state_dict()에 포함되어야 함
        # 따라서 일반 self.pe = pe가 아니라, register_buffer()를 사용해야 함.
        # 학습 중에는 gradient 계산 안 하고, .cuda(), .cpu(), .to() 같은 연산에서 자동으로 따라가며, 저장/불러오기 (.state_dict())도 정상적으로 작동함.
        # --> 학습은 하지 않지만, 모델 저장·로드, device 이동에는 포함되게 하고 싶다

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        #shape 맞추기 위해 self.pe에서 앞쪽 시퀀스만 잘라 씀
        # 원래 register_buffer로도 grad는 안 계산되지만, 이 코드는 실수로 autog   rad를 따라가지 않도록 확실하게 끊어주는 보강 조치라고 보면 됨.
        # 위에서 self.register_buffer('pe', pe) 했기 때문에 안해도 됨.
        return self.dropout(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout : float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1, B1 (512 --> 2048)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2, B2 (2048 --> 512)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model : int, h : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False) #wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) #wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) #wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) #wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout = nn.Dropout) -> Tuple[Tensor, Tensor] :
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None :
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # negative inf.
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)

        if dropout is not None :
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> # (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> # (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> # (Batch, seq_len, d_model)

        #query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        #key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        #value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = rearrange(query, 'batch seq_len (h d_k) -> batch h seq_len d_k', h=self.h)
        key = rearrange(key, 'batch seq_len (h d_k) -> batch h seq_len d_k', h=self.h)
        value = rearrange(value, 'batch seq_len (h d_k) -> batch h seq_len d_k', h=self.h)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, seq_len, h, d_k) --> # (Batch,seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps : float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward (self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x-mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # sublayer (예: attention, FFN), 특히 깊은 네트워크일수록 Pre-Norm이 gradient vanishing/exploding 문제에 더 강하다.


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float) -> None:
        # 여기서 MultiHeadAttentionBlock 및 FeedForwardBlock 각각 초기화를 위해 필요한 파라미터들을 정의해주지 않았기 때문에, 미리 이전에 정의가 된 후 해당 Encoder가 호출되어야 한다.
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock,feed_forward_block : FeedForwardBlock, dropout : float) -> None:
        # 여기서 MultiHeadAttentionBlock 및 FeedForwardBlock 각각 초기화를 위해 필요한 파라미터들을 정의해주지 않았기 때문에, 미리 이전에 정의가 된 후 해당 Encoder가 호출되어야 한다.
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self,
                 encoder : Encoder,
                 decoder : Decoder,
                 src_embed : InputEmbeddings,
                 tgt_embed : InputEmbeddings,
                 src_pos : PositionalEmbeddings,
                 tgt_pos : PositionalEmbeddings,
                 projection_layer = ProjectionLayer,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(encoder_output, src_mask, tgt, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
        src_vocab_size : int,
        tgt_vocab_size : int,
        src_seq_len : int,
        tgt_seq_len : int,
        d_model : int = 512,
        N : int = 6,
        h : int = 8,
        dropout : float = 0.1,
        d_ff : int = 2048)-> Transformer:

    #Create the Embedding Vectors
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, src_vocab_size)

    #Create the Positional Encoding Vectors
    src_pos = PositionalEmbeddings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbeddings(d_model, tgt_seq_len, dropout)

    #Create the Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #Create the Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder,
                 decoder,
                 src_embed,
                 tgt_embed,
                 src_pos,
                 tgt_pos,
                 projection_layer)


    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer








