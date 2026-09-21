"""
Stage 3: train the bigram model on the TinyStories token stream.

The chain, once per step:

    token -> logits -> softmax -> cross-entropy loss -> backward -> Adam step

Batches are drawn by picking random positions i in the stream and using
stream[i] as the input and stream[i+1] as the target.

Run from the project root:
    venv/bin/python train.py                                   # 5,000 steps, batch 64
    venv/bin/python train.py --batch-size 512 --run-name batch512
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines import (count_bigram_losses, select_smoothing_k, top_next_tokens,
                       uniform_loss, unigram_losses)
from model.bigram import BigramModel

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_TOKENS = PROJECT_ROOT / "data" / "stage3_train_tokens.npy"
VAL_TOKENS = PROJECT_ROOT / "data" / "stage3_val_tokens.npy"
RESULTS_FILE = PROJECT_ROOT / "results" / "stage3_bigram.md"

VOCAB_SIZE = 4000
STEPS = 5_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 1337

EVAL_EVERY = 250        # how often to measure train/val loss properly
EVAL_BATCHES = 50       # batches averaged per measurement
EVAL_BATCH_SIZE = 512   # bigger than the training batch: steadier, nearly free


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_stream(path, device):
    """Load a uint16 token stream and move it to the device as int64 IDs."""
    return torch.from_numpy(np.load(path).astype(np.int64)).to(device)


def get_batch(stream, batch_size):
    """Random positions in the stream: x = stream[i], y = stream[i+1]."""
    i = torch.randint(0, len(stream) - 1, (batch_size,), device=stream.device)
    return stream[i], stream[i + 1]


@torch.no_grad()
def estimate_loss(model, stream, batches=EVAL_BATCHES, batch_size=EVAL_BATCH_SIZE):
    """Average loss over several batches — far less noisy than one batch."""
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(stream, batch_size)
        total += F.cross_entropy(model(x), y).item()
    model.train()
    return total / batches


def write_results(path, lines):
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_text("# Stage 3 — bigram model results\n\n"
                        "One section per training run, appended automatically by train.py.\n",
                        encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train the Stage 3 bigram model")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="auto", help="auto, mps or cpu")
    parser.add_argument("--run-name", default=None, help="names the log and checkpoint files")
    args = parser.parse_args()
    run_name = args.run_name or f"batch{args.batch_size}"

    torch.manual_seed(SEED)
    device = pick_device(args.device)
    train_stream = load_stream(TRAIN_TOKENS, device)
    val_stream = load_stream(VAL_TOKENS, device)

    model = BigramModel(VOCAB_SIZE).to(device)
    # Plain Adam, NOT AdamW: weight decay would shrink the rows of tokens that
    # a step never even looked at.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"run {run_name} | device {device.type} | {num_params:,} parameters")
    print(f"train {len(train_stream):,} tokens | val {len(val_stream):,} tokens")
    print(f"steps {args.steps:,} | batch {args.batch_size} | lr {args.lr} | seed {SEED}")
    print(f"one epoch = {len(train_stream) - 1:,} pairs; this run sees "
          f"{args.steps * args.batch_size:,} "
          f"({100 * args.steps * args.batch_size / (len(train_stream) - 1):.1f}% of an epoch)\n")

    log_path = PROJECT_ROOT / "logs" / f"stage3_{run_name}.csv"
    log_path.parent.mkdir(exist_ok=True)
    log_file = log_path.open("w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["step", "batch_loss", "train_loss", "val_loss", "elapsed_s"])

    start_train = estimate_loss(model, train_stream)
    start_val = estimate_loss(model, val_stream)
    print(f"step {0:>6} | train {start_train:.4f} | val {start_val:.4f} "
          f"| uniform reference log(4000) = {uniform_loss(VOCAB_SIZE):.4f}")
    log.writerow([0, "", f"{start_train:.6f}", f"{start_val:.6f}", "0.0"])

    start = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_stream, args.batch_size)
        loss = F.cross_entropy(model(x), y)        # softmax + NLL, averaged over the batch
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                            # gradient of softmax+CE is (p - y)
        optimizer.step()

        # Dense logging early on, so a bad learning rate shows up immediately.
        is_eval_step = step % EVAL_EVERY == 0 or step == args.steps
        if is_eval_step or (step <= 500 and step % 50 == 0):
            elapsed = time.time() - start
            batch_loss = loss.item()
            if is_eval_step:
                train_loss = estimate_loss(model, train_stream)
                val_loss = estimate_loss(model, val_stream)
                print(f"step {step:>6} | batch {batch_loss:.4f} | train {train_loss:.4f} "
                      f"| val {val_loss:.4f} | ppl {np.exp(val_loss):>7.1f} | {elapsed:5.1f}s")
                log.writerow([step, f"{batch_loss:.6f}", f"{train_loss:.6f}",
                              f"{val_loss:.6f}", f"{elapsed:.1f}"])
            else:
                print(f"step {step:>6} | batch {batch_loss:.4f} | {elapsed:5.1f}s")
                log.writerow([step, f"{batch_loss:.6f}", "", "", f"{elapsed:.1f}"])

    wall_clock = time.time() - start
    final_train = estimate_loss(model, train_stream)
    final_val = estimate_loss(model, val_stream)
    log_file.close()

    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"stage3_{run_name}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab_size": VOCAB_SIZE,
                "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr},
               checkpoint_path)

    # --- reference losses, computed on the same streams ---------------------
    print("\nComputing baselines (counting pairs, no training involved)...")
    train_np, val_np = np.load(TRAIN_TOKENS), np.load(VAL_TOKENS)
    uni_train, uni_val = unigram_losses(train_np, val_np, VOCAB_SIZE)
    # The smoothing constant is chosen on a slice held out of the TRAINING
    # stream. Choosing it on val would tune the baseline against the very data
    # it is then measured on, flattering it.
    best_k, k_rows = select_smoothing_k(train_np, VOCAB_SIZE)
    mle_train, best_k_train, best_k_val = count_bigram_losses(train_np, val_np, VOCAB_SIZE, best_k)

    print("\nsmoothing k, scored on a held-out 5% slice of the training stream:")
    for k, tune_loss in k_rows:
        print(f"   k={k:<6} {tune_loss:.4f}{'   <- chosen' if k == best_k else ''}")

    print(f"\n{'reference':<34} {'train':>8} {'val':>8}")
    print(f"{'uniform (log 4000)':<34} {uniform_loss(VOCAB_SIZE):>8.4f} {uniform_loss(VOCAB_SIZE):>8.4f}")
    print(f"{'unigram (no context)':<34} {uni_train:>8.4f} {uni_val:>8.4f}")
    print(f"{'count-based bigram, k=' + str(best_k):<34} {best_k_train:>8.4f} {best_k_val:>8.4f}")
    print(f"{'count-based bigram, unsmoothed':<34} {mle_train:>8.4f} {'inf':>8}")
    print(f"{'TRAINED MODEL':<34} {final_train:>8.4f} {final_val:>8.4f}")

    # --- tracked results summary -------------------------------------------
    write_results(RESULTS_FILE, [
        f"## {run_name} — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- config: {args.steps:,} steps, batch {args.batch_size}, lr {args.lr}, "
        f"plain Adam, zero-init, seed {SEED}, device `{device.type}`",
        f"- data: {len(train_stream):,} train tokens / {len(val_stream):,} val tokens "
        f"(95/5 split by story, BPE vocab {VOCAB_SIZE:,})",
        f"- saw {args.steps * args.batch_size:,} pairs = "
        f"{100 * args.steps * args.batch_size / (len(train_stream) - 1):.1f}% of one epoch",
        f"- wall clock: {wall_clock:.1f}s ({1000 * wall_clock / args.steps:.2f} ms/step)",
        "",
        "| loss (nats) | train | val |",
        "|---|---|---|",
        f"| uniform, log(4000) | {uniform_loss(VOCAB_SIZE):.4f} | {uniform_loss(VOCAB_SIZE):.4f} |",
        f"| unigram (no context) | {uni_train:.4f} | {uni_val:.4f} |",
        f"| count-based bigram (add-{best_k}, k chosen on a held-out training slice) "
        f"| {best_k_train:.4f} | {best_k_val:.4f} |",
        f"| count-based bigram (unsmoothed) | {mle_train:.4f} | inf |",
        f"| **trained model** | **{final_train:.4f}** | **{final_val:.4f}** |",
        f"| trained model perplexity | {np.exp(final_train):.1f} | {np.exp(final_val):.1f} |",
        "",
        f"Log: `logs/stage3_{run_name}.csv` · checkpoint: `checkpoints/stage3_{run_name}.pt` "
        "(both gitignored)",
    ])
    print(f"\nSaved checkpoint -> {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print(f"Log -> {log_path.relative_to(PROJECT_ROOT)}")
    print(f"Results appended -> {RESULTS_FILE.relative_to(PROJECT_ROOT)}")

    # A quick look at what the model actually learned, next to raw counts.
    print("\nTop-5 next tokens after ' the' — model vs counts:")
    from tokenizer.bpe_tokenizer import BPETokenizer, SAVE_FILE
    tokenizer = BPETokenizer.load(SAVE_FILE)
    token_id = tokenizer.token_to_id[" the"]
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([token_id], device=device))[0], dim=-1)
    top = torch.topk(probs, 5)
    model_top = [(tokenizer.id_to_token[int(i)], float(p)) for p, i in zip(top.values, top.indices)]
    counts_top = [(tokenizer.id_to_token[i], p) for i, _, p in top_next_tokens(train_np, VOCAB_SIZE, token_id)]
    print(f"  model:  {', '.join(f'{t!r} {p:.3f}' for t, p in model_top)}")
    print(f"  counts: {', '.join(f'{t!r} {p:.3f}' for t, p in counts_top)}")


if __name__ == "__main__":
    main()
