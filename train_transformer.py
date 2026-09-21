"""
Stage 4 training foundations.

This module holds the pieces the LR sweep and the real training run share, so
both exercise identical code: batching, parameter groups, the clipped
optimizer step, and evaluation. The real run's main loop is added here once a
learning rate has been chosen.

Fixed parts of the training configuration (not variables under test):
  - AdamW, weight decay 0.01, applied ONLY to projection and FFN weights
  - global gradient-norm clipping at 1.0, before every optimizer step
  - batch 16, context 512, the Stage 3 95/5 train/monitoring split
"""

import argparse
import csv
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import CONTEXT_LENGTH, VOCAB_SIZE, TinyTransformer
from tokenizer.bpe_tokenizer import SAVE_FILE, BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_TOKENS = PROJECT_ROOT / "data" / "stage3_train_tokens.npy"
VAL_TOKENS = PROJECT_ROOT / "data" / "stage3_val_tokens.npy"
# data/tinystories_valid.txt stays untouched — it is reserved for later.
RESULTS_FILE = PROJECT_ROOT / "results" / "stage4_transformer.md"

BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 1337

STEPS = 5_000
LEARNING_RATE = 1e-3          # chosen from the lr_sweep.py evidence
EVAL_EVERY = 250
EVAL_BATCHES = 50             # fixed validation batches
CHECKPOINT_EVERY = 1_000
GENERATE_EVERY = 1_000        # only once the model is clearly learning
PROMPT = "Once there was a little girl"
SPIKE_RATIO = 10.0            # a norm this many times the median counts as a spike

# Stage 3 reference points, for the architectural comparison in the summary.
UNIFORM_LOSS = math.log(VOCAB_SIZE)        # 8.2940
UNIGRAM_VAL_LOSS = 6.0253
BIGRAM_VAL_LOSS = 4.6354                   # trained bigram, val perplexity 103.1
BIGRAM_COUNT_FLOOR_VAL = 3.7715            # best any bigram can do, val perplexity 43.5


def pick_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_stream(path, device):
    """Load a uint16 token stream and move it to the device as int64 IDs."""
    return torch.from_numpy(np.load(path).astype(np.int64)).to(device)


def _slice_batch(stream, starts, context_length):
    """Gather [B, T+1] windows, then split into inputs and targets.

    The one-token shift: a window of T+1 tokens becomes
        inputs  = window[:-1]   (stream[i   : i+T  ])
        targets = window[1:]    (stream[i+1 : i+T+1])
    so targets[:, t] == inputs[:, t+1]. Position t predicts what actually
    came next, never itself.
    """
    offsets = torch.arange(context_length + 1, device=stream.device)
    window = stream[starts[:, None] + offsets[None, :]]      # [B, T+1]
    return window[:, :-1], window[:, 1:]


def get_batch(stream, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH):
    """A random training batch: [B, T] inputs and [B, T] targets."""
    high = len(stream) - context_length - 1
    starts = torch.randint(0, high, (batch_size,), device=stream.device)
    return _slice_batch(stream, starts, context_length)


def make_fixed_batches(stream, num_batches, batch_size=BATCH_SIZE,
                       context_length=CONTEXT_LENGTH, seed=SEED):
    """
    A fixed, reproducible set of batches for evaluation.

    Using the same windows at every measurement means a change in the reported
    loss is a change in the model, not a change in which tokens got sampled.
    """
    rng = np.random.default_rng(seed)
    high = len(stream) - context_length - 1
    batches = []
    for _ in range(num_batches):
        starts = torch.from_numpy(rng.integers(0, high, size=batch_size)).to(stream.device)
        batches.append(_slice_batch(stream, starts, context_length))
    return batches


def make_param_groups(model, weight_decay=WEIGHT_DECAY):
    """
    Split parameters into decayed and undecayed groups.

    Weight decay pulls parameters toward zero, which makes sense for the
    projection and FFN weight matrices that do the mixing work. It does NOT
    make sense for:
      - token/position embeddings, where shrinking a rare token's row is just
        forgetting what little was learned about it,
      - LayerNorm weights, whose job is to rescale — decaying them toward 0
        fights the normalisation itself,
      - biases, which are offsets, not capacity.

    Returns (param_groups, rows) where rows is [(name, shape, numel, group)]
    for printing, so the split can be inspected instead of trusted.
    """
    decay_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            decay_names.add(f"{module_name}.weight")   # weights only, not biases

    decay, no_decay, rows = [], [], []
    for name, param in model.named_parameters():
        in_decay = name in decay_names
        (decay if in_decay else no_decay).append(param)
        rows.append((name, tuple(param.shape), param.numel(), "decay" if in_decay else "no_decay"))

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return groups, rows


def describe_param_groups(rows, weight_decay=WEIGHT_DECAY, show_all=False):
    """Print which named parameters landed in which group."""
    decay_rows = [r for r in rows if r[3] == "decay"]
    no_decay_rows = [r for r in rows if r[3] == "no_decay"]

    def summarise(title, subset, sample):
        total = sum(r[2] for r in subset)
        print(f"  {title}: {len(subset)} tensors, {total:,} parameters")
        for name, shape, numel, _ in (subset if show_all else sample):
            print(f"      {name:<34} {str(shape):<14} {numel:>9,}")
        if not show_all and len(subset) > len(sample):
            print(f"      ... and {len(subset) - len(sample)} more of the same pattern")
        return total

    print(f"AdamW parameter groups (weight_decay={weight_decay} / 0.0)")
    # Show every block-0 tensor plus anything outside the blocks: the other
    # five blocks repeat block 0 exactly.
    def block0_and_outside(subset):
        return [r for r in subset if not r[0].startswith("blocks.")
                or r[0].startswith("blocks.0.")]

    decayed = summarise("DECAYED", decay_rows, block0_and_outside(decay_rows))
    undecayed = summarise("NOT DECAYED", no_decay_rows, block0_and_outside(no_decay_rows))
    print(f"  total {decayed + undecayed:,} parameters "
          f"({100 * decayed / (decayed + undecayed):.1f}% decayed)")
    return decayed, undecayed


def train_step(model, optimizer, x, y, grad_clip=GRAD_CLIP):
    """
    One optimizer step. Returns (loss, pre-clip gradient norm).

    clip_grad_norm_ returns the norm BEFORE clipping, which is what tells us
    whether clipping actually intervened on this step.
    """
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item(), grad_norm.item()


@torch.no_grad()
def evaluate(model, batches):
    """Mean loss over a fixed list of (inputs, targets) batches."""
    model.eval()
    total = 0.0
    for x, y in batches:
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / len(batches)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60, temperature=1.0, device=None):
    """
    Sample a continuation, one token at a time.

    Each new token is drawn from the model's own distribution (temperature 1.0
    = no sharpening), appended, and fed back in. A qualitative check only.
    """
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    for _ in range(max_new_tokens):
        window = idx[:, -CONTEXT_LENGTH:]           # never exceed the context
        logits, _ = model(window)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
        if next_id.item() == tokenizer.eos_id:
            break
    model.train()
    return tokenizer.decode(idx[0].tolist())


def grad_norm_report(norms, ratio=SPIKE_RATIO):
    """Find isolated gradient spikes — the pattern clipping hides from the loss."""
    median = statistics.median(norms)
    spikes = [(i + 1, n) for i, n in enumerate(norms) if n > ratio * median]
    return median, max(norms), spikes


def save_checkpoint(path, model, optimizer, step, best_val, config):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "best_val_loss": best_val, "config": config}, path)


def main():
    parser = argparse.ArgumentParser(description="Stage 4 transformer training run")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="transformer")
    parser.add_argument("--resume", default=None,
                        help="checkpoint to continue from (model AND optimizer state)")
    parser.add_argument("--log-name", default=None, help="override the training CSV filename")
    parser.add_argument("--grad-name", default=None, help="override the grad-norm CSV filename")
    parser.add_argument("--samples-name", default=None, help="override the samples filename")
    parser.add_argument("--results-file", default=None, help="override the results summary path")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = pick_device(args.device)
    train_stream = load_stream(TRAIN_TOKENS, device)
    val_stream = load_stream(VAL_TOKENS, device)
    val_batches = make_fixed_batches(val_stream, EVAL_BATCHES, args.batch_size)

    model = TinyTransformer().to(device)
    groups, rows = make_param_groups(model, WEIGHT_DECAY)
    describe_param_groups(rows, WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    tokenizer = BPETokenizer.load(SAVE_FILE)

    tokens_per_step = args.batch_size * CONTEXT_LENGTH

    # --- resume ------------------------------------------------------------
    start_step, resumed_best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(PROJECT_ROOT / args.resume, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        # Loading the optimizer state is what makes this a continuation rather
        # than a fresh run at a higher step number: AdamW's first and second
        # moment estimates carry over instead of being rebuilt from zero.
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        resumed_best = checkpoint["best_val_loss"]
        # Keep the same seed, but fast-forward the batch RNG past the draws the
        # earlier run already made, so this run sees fresh windows instead of
        # replaying the same ones.
        high = len(train_stream) - CONTEXT_LENGTH - 1
        for _ in range(start_step):
            torch.randint(0, high, (args.batch_size,), device=device)
        print(f"resumed {args.resume} at step {start_step:,} "
              f"(best val {resumed_best:.4f}); optimizer state restored, "
              f"batch RNG fast-forwarded {start_step:,} draws")

    total_steps = start_step + args.steps
    config = {"steps": total_steps, "new_steps": args.steps, "resumed_from": args.resume,
              "batch_size": args.batch_size, "lr": args.lr,
              "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "seed": SEED,
              "context_length": CONTEXT_LENGTH, "vocab_size": VOCAB_SIZE}

    print(f"\nrun {args.run_name} | device {device.type} | {model.num_parameters():,} parameters")
    print(f"train {len(train_stream):,} tokens | val {len(val_stream):,} tokens "
          f"({EVAL_BATCHES} fixed eval batches)")
    print(f"steps {start_step + 1:,}-{total_steps:,} ({args.steps:,} new) | "
          f"batch {args.batch_size} | lr {args.lr} | clip {GRAD_CLIP} "
          f"| {tokens_per_step:,} tokens/step")
    print(f"one epoch = {len(train_stream):,} tokens; by step {total_steps:,} the model will "
          f"have seen {total_steps * tokens_per_step:,} "
          f"({total_steps * tokens_per_step / len(train_stream):.1f} epochs)\n")

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    log_path = logs_dir / (args.log_name or f"{args.run_name}_training.csv")
    grad_path = logs_dir / (args.grad_name or f"{args.run_name}_grad_norms.csv")
    samples_path = logs_dir / (args.samples_name or f"{args.run_name}_samples.txt")
    results_path = PROJECT_ROOT / args.results_file if args.results_file else RESULTS_FILE
    for path in (log_path, grad_path, samples_path):
        assert not path.exists(), f"refusing to overwrite {path}"

    log_file = log_path.open("w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["step", "train_loss", "val_loss", "val_ppl", "elapsed_seconds",
                  "tokens_seen", "grad_norm_max", "grad_norm_median", "clip_rate"])

    all_norms, interval_losses, curve = [], [], []
    best_val = resumed_best
    samples = []
    start = time.time()

    for step in range(start_step + 1, total_steps + 1):
        x, y = get_batch(train_stream, args.batch_size)
        loss, grad_norm = train_step(model, optimizer, x, y, GRAD_CLIP)
        interval_losses.append(loss)
        all_norms.append(grad_norm)

        if step % EVAL_EVERY == 0 or step == total_steps:
            val_loss = evaluate(model, val_batches)
            train_loss = sum(interval_losses) / len(interval_losses)
            window = all_norms[-len(interval_losses):]
            clip_rate = sum(1 for g in window if g > GRAD_CLIP) / len(window)
            elapsed = time.time() - start
            tokens_seen = step * tokens_per_step
            curve.append((step, train_loss, val_loss))

            log.writerow([step, f"{train_loss:.6f}", f"{val_loss:.6f}",
                          f"{math.exp(val_loss):.3f}", f"{elapsed:.1f}", tokens_seen,
                          f"{max(window):.4f}", f"{statistics.median(window):.4f}",
                          f"{clip_rate:.4f}"])
            log_file.flush()
            print(f"step {step:>5} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| ppl {math.exp(val_loss):>7.2f} | |g| med {statistics.median(window):.2f} "
                  f"max {max(window):.2f} | clip {100 * clip_rate:>4.1f}% | {elapsed / 60:4.1f}m")
            interval_losses = []

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(checkpoints_dir / f"{args.run_name}_best.pt",
                                model, optimizer, step, best_val, config)

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoints_dir / f"{args.run_name}_step{step}.pt",
                            model, optimizer, step, best_val, config)

        if step % GENERATE_EVERY == 0:
            text = generate(model, tokenizer, PROMPT, device=device)
            samples.append((step, text))
            print(f"\n  sample @ step {step}: {text!r}\n")

    wall_clock = time.time() - start
    final_train, final_val = curve[-1][1], curve[-1][2]
    log_file.close()

    with grad_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "grad_norm"])
        writer.writerows([[i + 1, f"{n:.6f}"] for i, n in enumerate(all_norms)])
    samples_path.write_text(
        "\n\n".join(f"=== step {s} ===\n{t}" for s, t in samples), encoding="utf-8")

    median_norm, max_norm, spikes = grad_norm_report(all_norms)
    overall_clip = sum(1 for g in all_norms if g > GRAD_CLIP) / len(all_norms)

    print(f"\ngradient norms over all {len(all_norms):,} steps: median {median_norm:.3f}, "
          f"max {max_norm:.3f} ({max_norm / median_norm:.1f}x median), "
          f"clipped on {100 * overall_clip:.1f}% of steps")
    if spikes:
        print(f"SPIKES above {SPIKE_RATIO}x median: {len(spikes)} "
              f"-> {[(s, round(n, 2)) for s, n in spikes[:10]]}")
    else:
        print(f"No step exceeded {SPIKE_RATIO}x the median — no sign of the 3e-3 pattern.")

    write_summary(args, config, wall_clock, curve, final_train, final_val, best_val,
                  median_norm, max_norm, spikes, overall_clip, samples,
                  tokens_per_step, len(train_stream), log_path, grad_path, samples_path,
                  checkpoints_dir, device, results_path)
    print(f"\nResults -> {results_path.relative_to(PROJECT_ROOT)}")


def write_summary(args, config, wall_clock, curve, final_train, final_val, best_val,
                  median_norm, max_norm, spikes, overall_clip, samples,
                  tokens_per_step, train_tokens, log_path, grad_path, samples_path,
                  checkpoints_dir, device, results_path):
    total_steps = config["steps"]
    tokens_seen = total_steps * tokens_per_step
    resumed = (f", resumed from `{config['resumed_from']}` at step "
               f"{total_steps - config['new_steps']:,}" if config["resumed_from"] else "")
    spike_line = (f"**{len(spikes)} step(s) above {SPIKE_RATIO}x the median** — "
                  f"{', '.join(f'step {s} ({n:.1f})' for s, n in spikes[:10])}. This is the "
                  f"pattern the lr=3e-3 sweep run showed, where clipping hid it from the loss."
                  if spikes else
                  f"No step exceeded {SPIKE_RATIO}x the median, so nothing resembling the "
                  f"lr=3e-3 spike pattern (38.4x at step 12) appeared.")

    lines = [
        "# Stage 4 — transformer results",
        "",
        f"## {args.run_name} — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- model: {config['context_length']}-token context, d_model 128, 4 heads, d_ff 512, "
        f"6 Pre-LN blocks, tied LM head — 1,767,424 parameters",
        f"- config: {config['new_steps']:,} steps this run (through step "
        f"{total_steps:,}){resumed}, batch {args.batch_size}, lr {args.lr}, AdamW "
        f"(weight decay {config['weight_decay']} on projection/FFN weights only), "
        f"grad clip {config['grad_clip']}, seed {config['seed']}, device `{device.type}`",
        f"- data: Stage 3's 95/5 split — {train_tokens:,} train tokens, validation from the "
        f"monitoring split (`tinystories_valid.txt` untouched)",
        f"- saw {tokens_seen:,} tokens = {tokens_seen / train_tokens:.1f} epochs",
        f"- wall clock: {wall_clock / 60:.1f} min ({1000 * wall_clock / config['new_steps']:.0f} ms/step)",
        "",
        "| loss (nats) | train | val | val perplexity |",
        "|---|---|---|---|",
        f"| **final (step {total_steps:,})** | **{final_train:.4f}** | **{final_val:.4f}** "
        f"| **{math.exp(final_val):.1f}** |",
        f"| best val | — | {best_val:.4f} | {math.exp(best_val):.1f} |",
        "",
        "### Reference points",
        "",
        "Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to "
        f"{config['context_length']}. The comparison is architectural.",
        "",
        "| model | val loss | val perplexity |",
        "|---|---|---|",
        f"| uniform, log(4000) | {UNIFORM_LOSS:.4f} | {math.exp(UNIFORM_LOSS):.0f} |",
        f"| unigram (Stage 3) | {UNIGRAM_VAL_LOSS:.4f} | {math.exp(UNIGRAM_VAL_LOSS):.1f} |",
        f"| trained bigram (Stage 3, best run) | {BIGRAM_VAL_LOSS:.4f} "
        f"| {math.exp(BIGRAM_VAL_LOSS):.1f} |",
        f"| count-based bigram floor (Stage 3) | {BIGRAM_COUNT_FLOOR_VAL:.4f} "
        f"| {math.exp(BIGRAM_COUNT_FLOOR_VAL):.1f} |",
        f"| **this transformer** | **{final_val:.4f}** | **{math.exp(final_val):.1f}** |",
        "",
        "### Gradient norms (pre-clip, all steps)",
        "",
        f"- median {median_norm:.3f}, max {max_norm:.3f} ({max_norm / median_norm:.1f}x median), "
        f"clipped on {100 * overall_clip:.1f}% of steps",
        f"- {spike_line}",
        "",
        "### Loss curve",
        "",
        "| step | train | val | val ppl |",
        "|---|---|---|---|",
    ]
    for step, train_loss, val_loss in curve:
        lines.append(f"| {step:,} | {train_loss:.4f} | {val_loss:.4f} | {math.exp(val_loss):.1f} |")

    # Is the rate of improvement holding, decelerating, or flattening?
    marks = [(s, v) for s, _, v in curve if s % 1000 == 0]
    if len(marks) > 1:
        lines += ["", "### Validation improvement per 1,000 steps", "",
                  "| segment | val loss | improvement | ppl |", "|---|---|---|---|"]
        for (prev_step, prev_val), (step, val) in zip(marks, marks[1:]):
            lines.append(f"| {prev_step:,} -> {step:,} | {val:.4f} | "
                         f"-{prev_val - val:.4f} | {math.exp(val):.2f} |")

    if samples:
        lines += ["", f"### Samples from \"{PROMPT}\" (temperature 1.0, qualitative only)", ""]
        for step, text in samples:
            lines += [f"**step {step:,}**", "", "```", text, "```", ""]

    lines += [
        f"Logs: `{log_path.name}`, `{grad_path.name}`, `{samples_path.name}` in `logs/` · "
        f"checkpoints: `{args.run_name}_best.pt` + periodic in `{checkpoints_dir.name}/` "
        "(both gitignored)",
    ]
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
