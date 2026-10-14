"""Small decoder-only, character-level GPT for teaching.

Shapes: tokens [B,T] -> logits [B,T,V]. No pretrained weights.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split_heads(a):
            return a.reshape(b, t, self.heads, c // self.heads).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))  # [B,H,T,D]
        scores = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().reshape(b, t, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, context: int = 16,
                 width: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.context = context
        self.token_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens, targets=None):
        b, t = tokens.shape
        if t > self.context:
            raise ValueError(f"sequence length {t} exceeds context {self.context}")
        positions = torch.arange(t, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        logits = self.head(self.norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, steps: int):
        for _ in range(steps):
            logits, _ = self(tokens[:, -self.context:])
            next_token = torch.multinomial(
                F.softmax(logits[:, -1, :], dim=-1), num_samples=1
            )
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
