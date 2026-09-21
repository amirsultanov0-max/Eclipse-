"""
Stage 4, final evaluation: the trained transformer on the OFFICIAL validation
file, which has been reserved and untouched since Stage 1.

Read-only. No training, no gradients, no sampling — one deterministic pass in
which every token is scored exactly once.

  1. Encode data/tinystories_valid.txt with the Stage 2b BPE tokenizer, one
     <EOS> per story, exactly as prepare_data.py builds the training stream.
  2. Cut the stream into non-overlapping windows of T=512, dropping the final
     partial window.
  3. Score every window under checkpoints/transformer_best.pt.
  4. Compute the Stage 3 baselines (uniform, unigram, count-based bigram at the
     already-fixed k=0.01) on this same file, so every number in the final
     table comes from identical data.

    venv/bin/python eval_official_valid.py
"""

import argparse
import math
from datetime import datetime

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines import _pair_counts
from model.bigram import BigramModel
from model.transformer import CONTEXT_LENGTH, VOCAB_SIZE, TinyTransformer
from tokenizer.bpe_tokenizer import SAVE_FILE, STORY_SEPARATOR, BPETokenizer
from train_transformer import PROJECT_ROOT, RESULTS_FILE, TRAIN_TOKENS, pick_device

VALID_FILE = PROJECT_ROOT / "data" / "tinystories_valid.txt"
DEFAULT_CHECKPOINT = "checkpoints/transformer_best.pt"
# Already recorded in results/stage4_transformer.md, carried into later tables so
# every transformer run can be compared on this same file.
TRANSFORMER_5K_OFFICIAL = 2.3236
# Stage 3's best bigram run (batch 64, lr 0.1), re-scored here on this same file.
BIGRAM_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "stage3_batch64_lr0.1.pt"
SMOOTHING_K = 0.01          # fixed in Stage 3, chosen on a held-out TRAINING slice
EVAL_BATCH = 16
POSITION_BUCKETS = [(1, 16), (17, 64), (65, 128), (129, 256), (257, 511)]


def encode_official_valid(tokenizer):
    """Same construction as prepare_data.py: story ... <EOS> story ... <EOS>"""
    text = VALID_FILE.read_text(encoding="utf-8")
    stories = [s.strip("\n") for s in text.split(STORY_SEPARATOR)]
    stories = [s for s in stories if s]
    ids = []
    for story in stories:
        ids.extend(tokenizer.encode(story))
        ids.append(tokenizer.eos_id)
    return np.array(ids, dtype=np.int64), len(stories)


def make_windows(stream, window_len):
    """Non-overlapping windows; the trailing partial window is dropped."""
    count = len(stream) // window_len
    return stream[:count * window_len].reshape(count, window_len)


@torch.no_grad()
def score_windows(model, windows, device, track_positions=False):
    """
    Total cross-entropy over every (input, target) pair in every window.

    Returns (total_loss_nats, token_count, per_position_sums). Summing rather
    than averaging batch means keeps the result exact regardless of batching.
    """
    total, tokens = 0.0, 0
    per_position = torch.zeros(windows.shape[1] - 1, device=device) if track_positions else None

    for start in range(0, len(windows), EVAL_BATCH):
        chunk = torch.from_numpy(windows[start:start + EVAL_BATCH]).to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = model(x)
        losses = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1),
                                 reduction="none")
        total += losses.sum().item()
        tokens += y.numel()
        if track_positions:
            per_position += losses.view(y.shape).sum(dim=0)
    return total, tokens, per_position


def baseline_losses(train_stream, inputs, targets):
    """
    Stage 3's baselines, measured on exactly the pairs the transformer scored.

    Counts come from the TRAINING stream (that is what the bigram was fitted
    on); only the evaluation data changes. The add-k formula is the one in
    baselines._add_k_loss, applied to a flat list of pairs instead of a stream.
    """
    counts, row_totals = _pair_counts(train_stream, VOCAB_SIZE)

    unigram_counts = np.bincount(train_stream, minlength=VOCAB_SIZE).astype(np.float64)
    unigram_logp = np.log((unigram_counts + 1.0) / (unigram_counts.sum() + VOCAB_SIZE))
    unigram = float(-unigram_logp[targets].mean())

    pair_counts = counts[inputs * VOCAB_SIZE + targets].astype(np.float64)
    totals = row_totals[inputs].astype(np.float64)
    probs = (pair_counts + SMOOTHING_K) / (totals + SMOOTHING_K * VOCAB_SIZE)
    bigram = float(-np.log(probs).mean())
    return unigram, bigram


@torch.no_grad()
def score_trained_bigram(inputs, targets, device):
    """
    Re-score Stage 3's trained bigram on the same pairs.

    The bigram's whole prediction for token x is row x, so one log_softmax over
    the 4,000 x 4,000 table gives every log-probability we need; after that it
    is a lookup, not a forward pass per token.
    """
    checkpoint = torch.load(BIGRAM_CHECKPOINT, map_location=device, weights_only=False)
    model = BigramModel(VOCAB_SIZE).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    log_probs = F.log_softmax(model.logits.weight, dim=-1)        # [V, V]

    total, step = 0.0, 500_000
    for start in range(0, len(inputs), step):
        x = torch.from_numpy(inputs[start:start + step]).to(device)
        y = torch.from_numpy(targets[start:start + step]).to(device)
        total += -log_probs[x, y].sum().item()
    return total / len(inputs), checkpoint


def main():
    parser = argparse.ArgumentParser(description="Official validation-set evaluation")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results-file", default=None,
                        help="results summary to APPEND to (default: the 5k results file)")
    args = parser.parse_args()
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    results_path = PROJECT_ROOT / args.results_file if args.results_file else RESULTS_FILE
    device = pick_device()
    tokenizer = BPETokenizer.load(SAVE_FILE)

    print(f"Encoding {VALID_FILE.name} (reserved since Stage 1, first use)...")
    stream, num_stories = encode_official_valid(tokenizer)
    windows = make_windows(stream, CONTEXT_LENGTH)
    dropped = len(stream) - windows.size
    print(f"  {num_stories:,} stories -> {len(stream):,} tokens")
    print(f"  {len(windows):,} full windows of {CONTEXT_LENGTH}; "
          f"dropped {dropped} tokens in the final partial window")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TinyTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"  checkpoint: step {checkpoint['step']:,}, "
          f"monitoring-split val loss {checkpoint['best_val_loss']:.4f}\n")

    total, tokens, per_position = score_windows(model, windows, device, track_positions=True)
    loss = total / tokens
    print(f"FULL PASS: {tokens:,} tokens scored exactly once "
          f"({100 * tokens / len(stream):.2f}% of the file's tokens)")
    print(f"  loss {loss:.4f} | perplexity {math.exp(loss):.2f}\n")

    # Cross-check: stride 512 over 513-token windows scores the tokens that the
    # non-overlapping windows skip (each window's first token).
    count = (len(stream) - 1) // CONTEXT_LENGTH
    strided = np.stack([stream[i * CONTEXT_LENGTH:i * CONTEXT_LENGTH + CONTEXT_LENGTH + 1]
                        for i in range(count)])
    strided_total, strided_tokens, _ = score_windows(model, strided, device)
    strided_loss = strided_total / strided_tokens
    print(f"  cross-check (every token scored, stride {CONTEXT_LENGTH}): "
          f"{strided_tokens:,} tokens, loss {strided_loss:.4f}, "
          f"ppl {math.exp(strided_loss):.2f}\n")

    # What is the 512-token context actually worth, by position in the window?
    print("Loss by position in the window (how much context the model has):")
    counts_per_position = len(windows)
    for lo, hi in POSITION_BUCKETS:
        bucket = per_position[lo - 1:hi].sum().item() / (counts_per_position * (hi - lo + 1))
        print(f"  targets at positions {lo:>3}-{hi:<3} (context {lo}-{hi} tokens): "
              f"loss {bucket:.4f} | ppl {math.exp(bucket):>6.2f}")

    inputs = windows[:, :-1].ravel()
    targets = windows[:, 1:].ravel()
    print("\nBaselines on the same pairs (counts fitted on the training stream)...")
    train_stream = np.load(TRAIN_TOKENS).astype(np.int64)
    unigram, bigram = baseline_losses(train_stream, inputs, targets)
    trained_bigram, bigram_ckpt = score_trained_bigram(inputs, targets, device)
    print(f"  Stage 3 bigram checkpoint: {BIGRAM_CHECKPOINT.name}, "
          f"{bigram_ckpt['steps']:,} steps at lr {bigram_ckpt['lr']}")

    uniform = math.log(VOCAB_SIZE)
    rows = [
        ("uniform, log(4000)", uniform),
        ("unigram (no context)", unigram),
        ("Stage 3 trained bigram (1-token context)", trained_bigram),
        ("count-based bigram floor (add-0.01)", bigram),
    ]
    if checkpoint_path.name != Path(DEFAULT_CHECKPOINT).name:
        # Keep the earlier transformer in the table, measured the same way.
        rows.append(("5k transformer, step 5,000 (recorded)", TRANSFORMER_5K_OFFICIAL))
    rows.append((f"this transformer, step {checkpoint['step']:,} (512-token context)", loss))
    print(f"\n{'model':<44} {'loss':>8} {'perplexity':>12}")
    for name, value in rows:
        print(f"{name:<44} {value:>8.4f} {math.exp(value):>12.2f}")

    append_summary(rows, loss, strided_loss, tokens, strided_tokens, len(stream),
                   len(windows), dropped, num_stories, per_position, checkpoint,
                   checkpoint_path, results_path)
    print(f"\nAppended to {results_path.relative_to(PROJECT_ROOT)}")


def append_summary(rows, loss, strided_loss, tokens, strided_tokens, stream_tokens,
                   num_windows, dropped, num_stories, per_position, checkpoint,
                   checkpoint_path, results_path):
    lines = [
        "",
        "---",
        "",
        f"## Official validation set — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Read-only evaluation of `{checkpoint_path.name}` "
        f"(step {checkpoint['step']:,}) on `data/tinystories_valid.txt`, reserved and "
        "untouched since Stage 1. One deterministic pass, every token scored exactly "
        "once — no sampling, no repeats.",
        "",
        f"- {num_stories:,} stories -> {stream_tokens:,} tokens "
        f"(BPE, one `<EOS>` per story, same construction as `prepare_data.py`)",
        f"- {num_windows:,} non-overlapping windows of {CONTEXT_LENGTH}; "
        f"{dropped} tokens dropped in the final partial window",
        f"- **{tokens:,} tokens scored** ({100 * tokens / stream_tokens:.2f}% of the file; "
        "each window's first token is an input only, never a target)",
        f"- cross-check at stride {CONTEXT_LENGTH} with 513-token windows, which scores every "
        f"token: {strided_tokens:,} tokens, loss {strided_loss:.4f} "
        f"(ppl {math.exp(strided_loss):.2f}) — the windowing choice is worth "
        f"{abs(strided_loss - loss):.4f} nats",
        "",
        "| model | loss (nats) | perplexity |",
        "|---|---|---|",
    ]
    for name, value in rows:
        bold = "**" if "transformer" in name else ""
        lines.append(f"| {bold}{name}{bold} | {bold}{value:.4f}{bold} "
                     f"| {bold}{math.exp(value):.2f}{bold} |")
    lines += [
        "",
        "Every row is measured on this same file, over the same token pairs. The Stage 3 "
        "bigram was re-scored from its own checkpoint rather than carried over. Bigram and "
        "unigram counts come from the training stream — only the evaluation data changed.",
        "",
        "This is an architectural comparison, not a like-for-like one: the bigram baselines "
        "see 1 token of context, the transformer up to 512.",
        "",
        "### Loss by position in the window",
        "",
        "| target positions | context available | loss | perplexity |",
        "|---|---|---|---|",
    ]
    for lo, hi in POSITION_BUCKETS:
        bucket = per_position[lo - 1:hi].sum().item() / (num_windows * (hi - lo + 1))
        lines.append(f"| {lo}-{hi} | {lo}-{hi} tokens | {bucket:.4f} "
                     f"| {math.exp(bucket):.2f} |")
    lines.append("")
    lines.append("Each window starts cold, so early positions are predicted from almost "
                 "nothing. This is what the reserved-file number costs relative to a sliding "
                 "window, and it is also a direct measure of what context is worth.")

    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
