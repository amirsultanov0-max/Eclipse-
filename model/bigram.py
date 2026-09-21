"""
Stage 3: the bigram next-token model.

The entire model is one learnable matrix of shape (vocab_size, vocab_size):
4,000 x 4,000 = 16,000,000 parameters. Row i holds the logits describing
"what token tends to follow token i".

There is no context beyond the single previous token, and nothing is shared
between rows — what the model learns about " the" tells it nothing about
" a". That limitation is the point: it is what the Transformer will fix.
"""

import torch.nn as nn


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # nn.Embedding IS the lookup table. Multiplying a one-hot vector by a
        # (vocab, vocab) matrix would give exactly the same result, but it
        # would spend 4,000 multiplications to fetch one row.
        self.logits = nn.Embedding(vocab_size, vocab_size)

        # Start every logit at zero. Softmax over all-zero logits is a uniform
        # distribution, so the very first loss is exactly log(vocab_size)
        # = 8.294 for 4,000 tokens — the "knows nothing" reference point.
        # (PyTorch's default init draws from N(0, 1), which starts ~8.8: random
        # logits are actively worse than admitting you know nothing.)
        nn.init.zeros_(self.logits.weight)

    def forward(self, idx):
        """idx: (B,) token IDs  ->  (B, vocab_size) logits, one row per input."""
        return self.logits(idx)
