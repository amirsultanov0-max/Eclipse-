"""
Stage 3 sanity checks — run these before spending a real training run.

  1. Step-0 loss must equal log(4000) = 8.2940 exactly. That confirms
     zero-init really does produce a uniform distribution.
  2. Overfit a tiny slice of 1,000 tokens. The loss must fall far below 8.294
     and approach the count-based floor for that slice. That confirms the
     whole chain (logits -> softmax -> cross-entropy -> backward -> Adam)
     actually moves the parameters in the right direction.
  3. Spot check: the overfitted model's top-5 predictions should match the
     raw pair counts of the slice it memorised.

Helpers are imported from train.py so both run exactly the same code.

    venv/bin/python sanity_check.py
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from baselines import count_bigram_losses, top_next_tokens
from model.bigram import BigramModel
from tokenizer.bpe_tokenizer import BPETokenizer, SAVE_FILE
from train import (SEED, TRAIN_TOKENS, VOCAB_SIZE, estimate_loss, get_batch,
                   load_stream, pick_device)

SLICE_TOKENS = 1_000
SLICE_STEPS = 2_000
SLICE_BATCH = 64
LEARNING_RATE = 1e-3


def main():
    torch.manual_seed(SEED)
    device = pick_device("auto")
    tokenizer = BPETokenizer.load(SAVE_FILE)
    stream = load_stream(TRAIN_TOKENS, device)
    print(f"device {device.type} | train stream {len(stream):,} tokens\n")

    # --- 1. step-0 loss ----------------------------------------------------
    print("1. Loss before any training")
    model = BigramModel(VOCAB_SIZE).to(device)
    measured = estimate_loss(model, stream)
    expected = math.log(VOCAB_SIZE)
    print(f"   measured {measured:.4f} | expected log(4000) = {expected:.4f} "
          f"| difference {abs(measured - expected):.2e}")
    print(f"   PASS: {abs(measured - expected) < 1e-3}\n")

    # --- 2. overfit a tiny slice -------------------------------------------
    print(f"2. Overfitting a {SLICE_TOKENS:,}-token slice for {SLICE_STEPS:,} steps")
    slice_np = np.load(TRAIN_TOKENS)[:SLICE_TOKENS]
    slice_stream = torch.from_numpy(slice_np.astype(np.int64)).to(device)

    # The floor: predict each next token using the counts of this slice alone.
    # Only the unsmoothed value (the first one) is used here, so k is irrelevant.
    floor, _, _ = count_bigram_losses(slice_np, slice_np, VOCAB_SIZE, k=0.01)
    print(f"   count-based floor for this slice: {floor:.4f}")

    model = BigramModel(VOCAB_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for step in range(1, SLICE_STEPS + 1):
        x, y = get_batch(slice_stream, SLICE_BATCH)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == 1:
            on_slice = estimate_loss(model, slice_stream, batches=20, batch_size=256)
            print(f"   step {step:>5} | loss on the slice {on_slice:.4f}")
    final = estimate_loss(model, slice_stream, batches=20, batch_size=256)
    fell = math.log(VOCAB_SIZE) - final
    print(f"   PASS (the loss really is being driven down): {fell > 0.5} "
          f"— fell {fell:.2f} nats, 8.2940 -> {final:.4f}")
    print(f"   reached the floor: {final < 1.2 * floor} "
          f"— {final:.4f} vs floor {floor:.4f}\n")

    # --- 3. top-5 spot check ------------------------------------------------
    print("3. Top-5 predictions of the overfitted model vs the slice's own counts")
    left_tokens, counts = np.unique(slice_np[:-1], return_counts=True)
    token_id = int(left_tokens[counts.argmax()])  # the most common token in the slice
    print(f"   after {tokenizer.id_to_token[token_id]!r} (appears {counts.max()} times in the slice)")

    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([token_id], device=device))[0], dim=-1)
    top = torch.topk(probs, 5)
    model_top = [(tokenizer.id_to_token[int(i)], float(p)) for p, i in zip(top.values, top.indices)]
    counts_top = [(tokenizer.id_to_token[i], share) for i, _, share in
                  top_next_tokens(slice_np, VOCAB_SIZE, token_id)]
    print(f"   model:  {', '.join(f'{t!r} {p:.3f}' for t, p in model_top)}")
    print(f"   counts: {', '.join(f'{t!r} {p:.3f}' for t, p in counts_top)}")
    print(f"   PASS: {[t for t, _ in model_top][:3] == [t for t, _ in counts_top][:3]} "
          "(top 3 in the same order)")


if __name__ == "__main__":
    main()
