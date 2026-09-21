"""
Stage 4, step 1: learning-rate sweep.

Four learning rates, everything else identical — same model init (same seed),
same batch size, same data order, same step count, same gradient clipping.
The only thing that differs between runs is the learning rate.

Reported per LR:
  - the loss curve, measured at fixed intervals on a FIXED set of batches, so
    the four curves are directly comparable
  - how often gradient clipping actually fired (pre-clip norm > 1.0). This
    matters because clipping hides instability: an LR too high produces huge
    gradients, and clipping quietly truncates them, so the loss curve can look
    calmer than the optimization really is.

This does not choose an LR. It reports evidence and proposes a region.

    venv/bin/python lr_sweep.py
"""

import time

import torch

from model.transformer import TinyTransformer
from train_transformer import (BATCH_SIZE, GRAD_CLIP, SEED, TRAIN_TOKENS, WEIGHT_DECAY,
                               describe_param_groups, evaluate, get_batch, load_stream,
                               make_fixed_batches, make_param_groups, pick_device, train_step)

LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
SWEEP_STEPS = 300
EVAL_EVERY = 25
EVAL_BATCHES = 10          # fixed batches, identical for every LR


def run_one(lr, stream, eval_batches, device):
    """Train a fresh model at this LR and return its curve and gradient stats."""
    torch.manual_seed(SEED)          # identical init and identical batch order
    model = TinyTransformer().to(device)
    groups, _ = make_param_groups(model, WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(groups, lr=lr)

    curve = [(0, evaluate(model, eval_batches))]
    grad_norms = []
    start = time.time()
    for step in range(1, SWEEP_STEPS + 1):
        x, y = get_batch(stream, BATCH_SIZE)
        _, grad_norm = train_step(model, optimizer, x, y, GRAD_CLIP)
        grad_norms.append(grad_norm)
        if step % EVAL_EVERY == 0:
            curve.append((step, evaluate(model, eval_batches)))
    elapsed = time.time() - start

    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return curve, grad_norms, elapsed


def classify(curve, grad_norms):
    """
    Describe the run using stated thresholds, rather than an impression.

      barely moving      total drop < 0.5 nats over the whole sweep
      early instability  any eval-to-eval RISE > 0.05 nats after step 50,
                         or a max gradient norm more than 10x the median
      smooth descent     neither of the above
    """
    losses = [loss for _, loss in curve]
    drop = losses[0] - losses[-1]
    rises = [(curve[i + 1][0], losses[i + 1] - losses[i])
             for i in range(len(losses) - 1) if curve[i + 1][0] > 50]
    worst_rise = max((r for _, r in rises), default=0.0)
    ordered = sorted(grad_norms)
    median_norm = ordered[len(ordered) // 2]
    spiky = max(grad_norms) > 10 * median_norm

    if worst_rise > 0.05 or spiky:
        verdict = "early instability"
    elif drop < 0.5:
        verdict = "barely moving"
    else:
        verdict = "smooth descent"
    return verdict, drop, worst_rise, median_norm


def main():
    device = pick_device()
    stream = load_stream(TRAIN_TOKENS, device)
    eval_batches = make_fixed_batches(stream, EVAL_BATCHES, BATCH_SIZE)

    # Verify the optimizer grouping before anything trains.
    reference = TinyTransformer().to(device)
    _, rows = make_param_groups(reference, WEIGHT_DECAY)
    describe_param_groups(rows, WEIGHT_DECAY)
    del reference
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"\nSweep: {len(LEARNING_RATES)} learning rates x {SWEEP_STEPS} steps | "
          f"batch {BATCH_SIZE} | clip {GRAD_CLIP} | seed {SEED} | device {device.type}")
    print(f"Loss measured every {EVAL_EVERY} steps on {EVAL_BATCHES} fixed batches "
          f"({EVAL_BATCHES * BATCH_SIZE * 512:,} tokens), identical across runs\n")

    results = {}
    for lr in LEARNING_RATES:
        curve, grad_norms, elapsed = run_one(lr, stream, eval_batches, device)
        results[lr] = (curve, grad_norms, elapsed)
        print(f"  lr={lr:<8} done in {elapsed:.0f}s   "
              f"loss {curve[0][1]:.4f} -> {curve[-1][1]:.4f}")

    # --- the four curves, side by side -------------------------------------
    print(f"\n{'step':>6} " + " ".join(f"{('lr=' + str(lr)):>12}" for lr in LEARNING_RATES))
    for i, (step, _) in enumerate(results[LEARNING_RATES[0]][0]):
        cells = " ".join(f"{results[lr][0][i][1]:>12.4f}" for lr in LEARNING_RATES)
        print(f"{step:>6} {cells}")

    # --- gradient clipping activity ----------------------------------------
    print(f"\n{'lr':>8} {'clipped':>9} {'first 50':>9} {'median |g|':>11} "
          f"{'max |g|':>9} {'verdict':<20}")
    for lr in LEARNING_RATES:
        curve, grad_norms, _ = results[lr]
        clipped = sum(1 for g in grad_norms if g > GRAD_CLIP)
        early = sum(1 for g in grad_norms[:50] if g > GRAD_CLIP)
        verdict, drop, worst_rise, median_norm = classify(curve, grad_norms)
        print(f"{lr:>8} {100 * clipped / len(grad_norms):>8.1f}% {100 * early / 50:>8.1f}% "
              f"{median_norm:>11.3f} {max(grad_norms):>9.2f} {verdict:<20}")

    print(f"\n{'lr':>8} {'start':>8} {'final':>8} {'drop':>8} {'worst rise after step 50':>26}")
    for lr in LEARNING_RATES:
        curve, grad_norms, _ = results[lr]
        _, drop, worst_rise, _ = classify(curve, grad_norms)
        print(f"{lr:>8} {curve[0][1]:>8.4f} {curve[-1][1]:>8.4f} {drop:>8.4f} {worst_rise:>26.4f}")


if __name__ == "__main__":
    main()
