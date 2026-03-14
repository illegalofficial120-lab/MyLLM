"""
GPT-style Transformer model built from scratch.
Components: embeddings, sinusoidal positional encoding, multi-head self-attention,
feed-forward network, transformer blocks, and language modeling head.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


def get_positional_encoding(max_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Sinusoidal positional encoding (Vaswani et al., "Attention is All You Need").
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    position = torch.arange(max_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


class Embedding(nn.Module):
    """Token embeddings + optional learned positional embeddings. We use fixed sinusoidal by default."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)
        # Register fixed sinusoidal PE (not a parameter)
        pe = get_positional_encoding(config.max_seq_length, config.d_model, device=torch.device("cpu"))
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        seq_len = x.size(1)
        tok_emb = self.token_embedding(x)  # (batch, seq_len, d_model)
        pos_emb = self.pe[:, :seq_len, :]
        out = tok_emb + pos_emb
        return self.dropout(out)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal (autoregressive) mask.
    Q, K, V are computed from the same input; we mask out future positions.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * self.n_heads == self.d_model
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, S, d)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        # Causal mask: mask out future positions
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if attention_mask is not None:
            # attention_mask: (B, S) or (B, 1, 1, S); 0 = mask out
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, S, d)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward: two linear layers with GELU in between (GPT-2 style)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm -> Self-Attention -> residual -> LayerNorm -> FFN -> residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), attention_mask=attention_mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TransformerLM(nn.Module):
    """
    Full GPT-style language model:
    Embedding -> N x TransformerBlock -> LayerNorm -> LM head (linear to vocab).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie weights: lm_head weight = embedding weight (optional, saves params and often helps)
        self.lm_head.weight = self.embedding.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: (batch, seq_len) 1 for valid, 0 for padding
            labels: (batch, seq_len) for loss; same as input_ids shifted by 1 (handled in trainer)
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar, only if labels is provided
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Cross-entropy: flatten and ignore padding (id 0 or mask)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous().view(-1)
                loss = F.cross_entropy(
                    shift_logits[shift_mask.bool()],
                    shift_labels[shift_mask.bool()],
                    ignore_index=-100,
                )
            else:
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
