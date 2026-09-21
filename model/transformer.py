"""
Stage 4: the transformer.

Unlike the Stage 3 bigram — one row of logits per token, no context — this
model looks at up to 512 previous tokens and builds each prediction from them.

Architecture (locked in during design):
    vocab 4,000 | context T=512 | d_model D=128 | heads H=4 (head_dim 32)
    d_ff F=512  | 6 blocks | Pre-LN | GELU | learned absolute positions
    causal masked attention | LM head tied to the token embedding

Every forward pass follows this shape contract exactly:

    [B, T]                      token IDs
    -> [B, T, D]                token embedding + position embedding
    per block:
       -> [B, T, D]             LayerNorm (Pre-LN: normalise the INPUT of
                                each sublayer, not the output)
       -> [B, T, D] x3          Q, K, V projections
       -> [B, H, T, Dh] x3      split into heads
       -> [B, H, T, T]          QK^T / sqrt(Dh)
       -> [B, H, T, T]          + causal mask (additive, before softmax)
       -> [B, H, T, T]          softmax over the last dim
       -> [B, H, T, Dh]         weights @ V
       -> [B, T, D]             merge heads
       -> [B, T, D]             output projection W_O
       -> [B, T, D]             residual add
       -> [B, T, D]             LayerNorm
       -> [B, T, F]             FFN first layer + GELU
       -> [B, T, D]             FFN second layer
       -> [B, T, D]             residual add
    -> [B, T, D]                final LayerNorm
    -> [B, T, V]                LM head (tied weights)

Pass `trace={}` to forward() to have every one of those stages recorded, which
is what shape_test.py checks against.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 4_000
CONTEXT_LENGTH = 512      # T
D_MODEL = 128             # D
NUM_HEADS = 4             # H  (head_dim Dh = D // H = 32)
D_FF = 512                # F
NUM_BLOCKS = 6
INIT_STD = 0.02           # small random init, GPT-2 style


def _record(trace, key, tensor):
    """Note a tensor's shape if tracing is on. A no-op during real training."""
    if trace is not None:
        trace[key] = tuple(tensor.shape)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where a position may only look backwards."""

    def __init__(self, d_model, num_heads, context_length):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly into heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Three separate projections, so the shapes stay obvious. (Production
        # code usually fuses them into one Linear(D, 3D) for speed.)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # The causal mask: 0 where attention is allowed, -inf where it is not.
        # Adding -inf before the softmax drives those weights to exactly zero.
        # triu(1) keeps everything strictly ABOVE the diagonal, i.e. the future.
        mask = torch.full((context_length, context_length), float("-inf")).triu(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, trace=None, prefix=""):
        B, T, D = x.shape

        q = self.q_proj(x)                                   # [B, T, D]
        k = self.k_proj(x)
        v = self.v_proj(x)
        _record(trace, prefix + "q_proj", q)
        _record(trace, prefix + "k_proj", k)
        _record(trace, prefix + "v_proj", v)

        # Split D into H heads of Dh, then put the head axis second so every
        # head is its own independent [T, Dh] attention problem.
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, T, Dh]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        _record(trace, prefix + "q_heads", q)
        _record(trace, prefix + "k_heads", k)
        _record(trace, prefix + "v_heads", v)

        # How much does each position want to read from each other position?
        # Dividing by sqrt(Dh) keeps the scores from growing with head_dim,
        # which would otherwise push softmax into a near-one-hot regime.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # [B, H, T, T]
        _record(trace, prefix + "scores", scores)

        scores = scores + self.causal_mask[:T, :T]           # broadcast over B and H
        _record(trace, prefix + "masked_scores", scores)

        weights = F.softmax(scores, dim=-1)                  # [B, H, T, T]
        _record(trace, prefix + "attn_weights", weights)

        attended = weights @ v                               # [B, H, T, Dh]
        _record(trace, prefix + "attn_output", attended)

        # Merge the heads back into one vector per position. contiguous() is
        # needed because transpose only changes how the tensor is viewed.
        merged = attended.transpose(1, 2).contiguous().view(B, T, D)      # [B, T, D]
        _record(trace, prefix + "merged_heads", merged)

        projected = self.out_proj(merged)                    # [B, T, D]
        _record(trace, prefix + "attn_out_proj", projected)
        return projected


class TransformerBlock(nn.Module):
    """Pre-LN block: x + Attention(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model, num_heads, d_ff, context_length):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, context_length)
        self.ln_2 = nn.LayerNorm(d_model)
        # The FFN is written out rather than wrapped in nn.Sequential so the
        # [B, T, F] hidden layer can be traced.
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)

    def forward(self, x, trace=None, prefix=""):
        # --- attention sublayer ---
        normed = self.ln_1(x)
        _record(trace, prefix + "ln_1", normed)
        x = x + self.attn(normed, trace, prefix)             # residual add
        _record(trace, prefix + "residual_1", x)

        # --- feed-forward sublayer ---
        normed = self.ln_2(x)
        _record(trace, prefix + "ln_2", normed)
        hidden = F.gelu(self.ff_in(normed))                  # [B, T, F]
        _record(trace, prefix + "ffn_hidden", hidden)
        ff = self.ff_out(hidden)                             # [B, T, D]
        _record(trace, prefix + "ffn_out", ff)
        x = x + ff                                           # residual add
        _record(trace, prefix + "residual_2", x)
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH,
                 d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, num_blocks=NUM_BLOCKS):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)      # [V, D]
        # Learned absolute positions: row t is "what it means to be at slot t".
        self.position_embedding = nn.Embedding(context_length, d_model)  # [T, D]

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length)
            for _ in range(num_blocks)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # No separate LM head: the logits come from the token embedding matrix,
        # transposed. "How much does this vector look like token j's embedding?"
        # That ties 512,000 parameters into double duty instead of duplicating
        # them, and it is why there is no self.lm_head here.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Small random init. LayerNorm keeps PyTorch's default (weight 1, bias 0)."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, trace=None):
        """
        idx:     [B, T] token IDs
        targets: [B, T] the same stream shifted left by one, targets[:, t] = idx[:, t+1]
        returns: logits [B, T, V] and (if targets given) the mean loss
        """
        B, T = idx.shape
        assert T <= self.context_length, f"sequence of {T} exceeds context {self.context_length}"

        token_emb = self.token_embedding(idx)                              # [B, T, D]
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(positions)                       # [T, D]
        _record(trace, "token_embedding", token_emb)
        _record(trace, "position_embedding", pos_emb)

        x = token_emb + pos_emb        # [T, D] broadcasts across the batch
        _record(trace, "embeddings_sum", x)

        for i, block in enumerate(self.blocks):
            x = block(x, trace, prefix=f"block{i}.")
        _record(trace, "blocks_output", x)

        x = self.ln_final(x)                                               # [B, T, D]
        _record(trace, "ln_final", x)

        # Tied LM head: [B, T, D] @ [D, V] -> [B, T, V]
        logits = x @ self.token_embedding.weight.T
        _record(trace, "logits", logits)

        loss = None
        if targets is not None:
            # cross_entropy wants a flat list of predictions and answers, so
            # every one of the B*T positions becomes its own training example.
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss
