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

Randomness (see main() for the full note):
  - --seed          model init and training-batch sampling (default 1337)
  - EVAL_SEED       the fixed monitoring-eval windows; never follows --seed
  - --seed + 10000  a dedicated generator for in-training samples only

Every run writes a provenance manifest next to its results file (git commit and
dirty flag, all arguments, seeds, SHA256 of the data streams, and fingerprints of
the initial weights, eval batches and every step's batch positions).

Learning-rate schedule (--lr-schedule):
  - constant        the default and every run before Stage 6.3: --lr throughout
  - warmup_cosine   the Stage 6.3 treatment (eval/stage6_3_preregistration.md 2.1):
                    linear warm-up to --lr over steps 1-1,000, then cosine decay to
                    1e-4 at step 10,000; a shorter run follows the same schedule

    venv/bin/python train_transformer.py --run-name NAME --results-file results/NAME.md
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import (CONTEXT_LENGTH, D_FF, D_MODEL, NUM_HEADS, VOCAB_SIZE,
                               TinyTransformer)
from tokenizer.bpe_tokenizer import SAVE_FILE, BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_TOKENS = PROJECT_ROOT / "data" / "stage3_train_tokens.npy"
VAL_TOKENS = PROJECT_ROOT / "data" / "stage3_val_tokens.npy"
# data/tinystories_valid.txt stays untouched — it is reserved for later.
RESULTS_FILE = PROJECT_ROOT / "results" / "stage4_transformer.md"

BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 1337                    # default for --seed: model init + training-batch draws
EVAL_SEED = 1337               # fixed: selects the monitoring-eval windows
GENERATION_SEED_OFFSET = 10_000   # in-training samples use their own generator

STEPS = 5_000
LEARNING_RATE = 1e-3          # chosen from the lr_sweep.py evidence
EVAL_EVERY = 250
EVAL_BATCHES = 50             # fixed validation batches
CHECKPOINT_EVERY = 1_000
GENERATE_EVERY = 1_000        # only once the model is clearly learning
PROMPT = "Once there was a little girl"
SPIKE_RATIO = 10.0            # a norm this many times the median counts as a spike

LR_SCHEDULES = ("constant", "warmup_cosine")
# The Stage 6.3 treatment schedule, fixed by eval/stage6_3_preregistration.md section 2.1.
WARMUP_STEPS = 1_000
DECAY_END_STEP = 10_000
FINAL_LR = 1e-4

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


def get_batch(stream, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH,
              return_starts=False):
    """
    A random training batch: [B, T] inputs and [B, T] targets.

    With return_starts=True the start positions are returned as well, for
    fingerprinting. The random draw is identical either way.
    """
    high = len(stream) - context_length - 1
    starts = torch.randint(0, high, (batch_size,), device=stream.device)
    x, y = _slice_batch(stream, starts, context_length)
    return (x, y, starts) if return_starts else (x, y)


def make_fixed_batches(stream, num_batches, batch_size=BATCH_SIZE,
                       context_length=CONTEXT_LENGTH, seed=EVAL_SEED):
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


def warmup_cosine_lr(step, peak):
    """
    The learning rate for the update of `step` (1-based) under the Stage 6.3 schedule.

    Written in the order of operations of eval/stage6_3_preregistration.md section 2.1,
    so an independent evaluation of that formula reproduces it bit for bit.
    """
    if step <= WARMUP_STEPS:
        return peak * step / WARMUP_STEPS
    return FINAL_LR + 0.5 * (peak - FINAL_LR) * (
        1 + math.cos(math.pi * (step - WARMUP_STEPS) / (DECAY_END_STEP - WARMUP_STEPS)))


def lr_schedule_config(args):
    """What the run's config and manifest record about its learning-rate schedule."""
    if args.lr_schedule == "constant":
        return {"name": "constant", "lr": args.lr}
    return {"name": "warmup_cosine", "peak_lr": args.lr, "warmup_steps": WARMUP_STEPS,
            "decay_end_step": DECAY_END_STEP, "final_lr": FINAL_LR,
            "defined_in": "eval/stage6_3_preregistration.md section 2.1"}


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
def generate(model, tokenizer, prompt, max_new_tokens=60, temperature=1.0, device=None,
             generator=None, return_draws=False):
    """
    Sample a continuation, one token at a time.

    Each new token is drawn from the model's own distribution (temperature 1.0
    = no sharpening), appended, and fed back in. A qualitative check only.

    generator=None draws from the device's default generator (the original
    behaviour). return_draws=True also returns how many random draws were made.
    """
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    draws = 0
    for _ in range(max_new_tokens):
        window = idx[:, -CONTEXT_LENGTH:]           # never exceed the context
        logits, _ = model(window)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        draws += 1
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
        if next_id.item() == tokenizer.eos_id:
            break
    model.train()
    text = tokenizer.decode(idx[0].tolist())
    return (text, draws) if return_draws else text


def grad_norm_report(norms, ratio=SPIKE_RATIO, first_step=1):
    """Find isolated gradient spikes — the pattern clipping hides from the loss."""
    median = statistics.median(norms)
    spikes = [(first_step + i, n) for i, n in enumerate(norms) if n > ratio * median]
    return median, max(norms), spikes


def display_path(path):
    """A path relative to the repo when it is inside it, absolute otherwise."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensors(named_tensors):
    """SHA256 over (name, raw bytes) of each tensor, in the order given. No RNG."""
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def git_provenance():
    """Commit hash, dirty flag, untracked files, and the uncommitted diff (if any)."""
    def git(*cmd):
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *cmd],
                              capture_output=True, text=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip() or "unknown"
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        untracked = [f for f in git("ls-files", "--others", "--exclude-standard").splitlines()]
        diff = git("diff", "HEAD") if dirty else ""
    except FileNotFoundError:
        return {"commit": "unknown", "dirty": None, "untracked": []}, ""
    return {"commit": commit, "dirty": dirty, "untracked": untracked}, diff


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
    parser.add_argument("--train-tokens", default=None,
                        help="override the training token stream (default: Stage 3's)")
    parser.add_argument("--log-name", default=None, help="override the training CSV filename")
    parser.add_argument("--grad-name", default=None, help="override the grad-norm CSV filename")
    parser.add_argument("--samples-name", default=None, help="override the samples filename")
    parser.add_argument("--results-file", required=True,
                        help="results summary to write; must not already exist")
    parser.add_argument("--results-title", default="Stage 4 — transformer results",
                        help="H1 heading of the results file (default keeps earlier files identical)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="model init and training-batch sampling (default 1337)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="where checkpoints go (relative to the repo, or absolute)")
    # Architecture. The defaults are model/transformer.py's own constants, so a run
    # that passes none of these builds exactly the model every earlier run built.
    parser.add_argument("--d-model", type=int, default=D_MODEL,
                        help=f"model width (default {D_MODEL})")
    parser.add_argument("--n-heads", type=int, default=NUM_HEADS,
                        help=f"attention heads (default {NUM_HEADS})")
    parser.add_argument("--d-ff", type=int, default=D_FF,
                        help=f"feed-forward width (default {D_FF})")
    # LR schedule. The default is the constant rate every earlier run used; under it
    # no learning rate is ever written after the optimizer is built.
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="constant",
                        help="constant (default) or warmup_cosine (Stage 6.3 treatment)")
    args = parser.parse_args()
    scheduled = args.lr_schedule == "warmup_cosine"
    if scheduled and args.resume:
        raise SystemExit("--lr-schedule warmup_cosine cannot be combined with --resume: "
                         "treatment runs are single passes from step 1")
    if scheduled and args.steps > DECAY_END_STEP:
        raise SystemExit(f"--lr-schedule warmup_cosine is defined for steps 1-{DECAY_END_STEP:,} "
                         f"only; got --steps {args.steps:,}")

    # --- output paths and provenance: pure I/O, no random draws -------------
    # Everything is checked before training starts, so a clash fails in seconds
    # rather than after a full run.
    train_path = PROJECT_ROOT / args.train_tokens if args.train_tokens else TRAIN_TOKENS
    logs_dir = PROJECT_ROOT / "logs"
    checkpoints_dir = PROJECT_ROOT / args.checkpoint_dir     # absolute paths pass through
    log_path = logs_dir / (args.log_name or f"{args.run_name}_training.csv")
    grad_path = logs_dir / (args.grad_name or f"{args.run_name}_grad_norms.csv")
    samples_path = logs_dir / (args.samples_name or f"{args.run_name}_samples.txt")
    results_path = PROJECT_ROOT / args.results_file         # absolute paths pass through
    manifest_path = results_path.with_name(results_path.stem + ".provenance.json")
    diff_path = results_path.with_name(results_path.stem + ".provenance.diff")
    # Per-step learning rates go in their own file, so the per-step log keeps its format.
    lr_path = grad_path.with_name(grad_path.stem + "_lr.csv")
    for path in (log_path, grad_path, samples_path, results_path, manifest_path, diff_path) + (
            (lr_path,) if scheduled else ()):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    for directory in {checkpoints_dir, log_path.parent, grad_path.parent,
                      samples_path.parent, results_path.parent}:
        directory.mkdir(parents=True, exist_ok=True)

    git_info, git_diff = git_provenance()
    if git_diff:
        diff_path.write_text(git_diff, encoding="utf-8")
    data_files = {"train_tokens": train_path, "val_tokens": VAL_TOKENS}
    provenance = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "git": {**git_info, "diff_file": display_path(diff_path) if git_diff else None},
        "argv": sys.argv,
        "args": vars(args),
        "seeds": {"seed": args.seed, "eval_seed": EVAL_SEED,
                  "generation_seed": args.seed + GENERATION_SEED_OFFSET,
                  "generation_rng": "dedicated torch.Generator (decoupled from batches)"},
        "data": {name: {"path": display_path(path), "sha256": sha256_file(path)}
                 for name, path in data_files.items()},
        "resume": ({"checkpoint": args.resume,
                    "sha256": sha256_file(PROJECT_ROOT / args.resume)}
                   if args.resume else None),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
        "lr_schedule": lr_schedule_config(args),
    }

    # --- randomness ---------------------------------------------------------
    # --seed drives the default generators: model init (CPU) and training-batch
    # sampling (the training device). EVAL_SEED alone picks the monitoring-eval
    # windows (numpy), so they are identical for every --seed.
    #
    # In-training samples (every GENERATE_EVERY steps) draw from their OWN
    # generator, seeded --seed + GENERATION_SEED_OFFSET. Before this change they
    # drew from the shared device generator, so each sample shifted every later
    # training batch — by a number of draws that depends on where the sample
    # happens to hit <EOS>. Now the batch stream is independent of generation.
    # With --seed 1337 every non-generation draw is identical to the pre-change
    # code; only the generation draws have left the shared stream, so samples
    # (and batches after the first sample) differ from pre-change runs.
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    generation_rng = torch.Generator(device=device)
    generation_rng.manual_seed(args.seed + GENERATION_SEED_OFFSET)
    train_stream = load_stream(train_path, device)
    val_stream = load_stream(VAL_TOKENS, device)   # monitoring split, never overridden
    val_batches = make_fixed_batches(val_stream, EVAL_BATCHES, args.batch_size,
                                     seed=EVAL_SEED)

    model = TinyTransformer(d_model=args.d_model, num_heads=args.n_heads,
                            d_ff=args.d_ff).to(device)
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
        # replaying the same ones. With generation on its own generator this
        # reproduces an uninterrupted run's batches exactly — except when
        # resuming a checkpoint written by the pre-change code, whose shared
        # stream also carried generation draws that are not replayed here. The
        # generation generator itself is not checkpointed: it restarts from its
        # seed, which affects only the qualitative samples.
        high = len(train_stream) - CONTEXT_LENGTH - 1
        for _ in range(start_step):
            torch.randint(0, high, (args.batch_size,), device=device)
        print(f"resumed {args.resume} at step {start_step:,} "
              f"(best val {resumed_best:.4f}); optimizer state restored, "
              f"batch RNG fast-forwarded {start_step:,} draws")

    # Fingerprints of what the loop starts from: no random draws involved.
    provenance["fingerprints"] = {
        "initial_weights_sha256": sha256_tensors(sorted(model.state_dict().items())),
        "eval_batches_sha256": sha256_tensors(
            (f"{i}.{part}", t) for i, (x, y) in enumerate(val_batches)
            for part, t in (("x", x), ("y", y))),
    }
    manifest_path.write_text(json.dumps(provenance, indent=1), encoding="utf-8")

    total_steps = start_step + args.steps
    config = {"steps": total_steps, "new_steps": args.steps, "resumed_from": args.resume,
              "train_tokens": display_path(train_path),
              "batch_size": args.batch_size, "lr": args.lr,
              "lr_schedule": lr_schedule_config(args),
              "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "seed": args.seed,
              "eval_seed": EVAL_SEED, "generation_seed": args.seed + GENERATION_SEED_OFFSET,
              "context_length": CONTEXT_LENGTH, "vocab_size": VOCAB_SIZE,
              "d_model": args.d_model, "n_heads": args.n_heads, "d_ff": args.d_ff,
              "num_blocks": len(model.blocks),
              "parameters": model.num_parameters(),
              "args": vars(args),
              "provenance": {"git": provenance["git"], "data": provenance["data"],
                             "fingerprints": provenance["fingerprints"],
                             "manifest": display_path(manifest_path)}}

    print(f"\nrun {args.run_name} | device {device.type} | {model.num_parameters():,} parameters")
    print(f"train {len(train_stream):,} tokens | val {len(val_stream):,} tokens "
          f"({EVAL_BATCHES} fixed eval batches)")
    print(f"steps {start_step + 1:,}-{total_steps:,} ({args.steps:,} new) | "
          f"batch {args.batch_size} | lr {args.lr}"
          f"{' (warmup_cosine)' if scheduled else ''} | clip {GRAD_CLIP} "
          f"| {tokens_per_step:,} tokens/step")
    print(f"one epoch = {len(train_stream):,} tokens; by step {total_steps:,} the model will "
          f"have seen {total_steps * tokens_per_step:,} "
          f"({total_steps * tokens_per_step / len(train_stream):.1f} epochs)\n")

    log_file = log_path.open("w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["step", "train_loss", "val_loss", "val_ppl", "elapsed_seconds",
                  "tokens_seen", "grad_norm_max", "grad_norm_median", "clip_rate"])

    all_norms, all_losses, batch_hashes, interval_losses, curve = [], [], [], [], []
    running_batch_hash = hashlib.sha256()      # over the concatenated per-step hashes
    best_val = resumed_best
    samples, generation_draws = [], {}
    step_lrs = []
    start = time.time()

    for step in range(start_step + 1, total_steps + 1):
        x, y, starts = get_batch(train_stream, args.batch_size, return_starts=True)
        batch_hash = hashlib.sha256(starts.cpu().numpy().tobytes()).hexdigest()
        running_batch_hash.update(bytes.fromhex(batch_hash))
        if scheduled:
            # Plain arithmetic, no random draws: the batch stream is unaffected.
            lr = warmup_cosine_lr(step, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            step_lrs.append((step, lr))
        loss, grad_norm = train_step(model, optimizer, x, y, GRAD_CLIP)
        interval_losses.append(loss)
        all_losses.append(loss)
        all_norms.append(grad_norm)
        batch_hashes.append(batch_hash)

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
            text, draws = generate(model, tokenizer, PROMPT, device=device,
                                   generator=generation_rng, return_draws=True)
            samples.append((step, text))
            generation_draws[step] = draws
            print(f"\n  sample @ step {step}: {text!r}\n")

    wall_clock = time.time() - start
    final_train, final_val = curve[-1][1], curve[-1][2]
    log_file.close()

    # Per-step log, numbered by the true global step (a resumed run starts at
    # start_step + 1), at full float precision, with each step's batch fingerprint.
    with grad_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "grad_norm", "batch_starts_sha256"])
        writer.writerows([[start_step + i + 1, repr(loss_i), repr(norm_i), hash_i]
                          for i, (loss_i, norm_i, hash_i)
                          in enumerate(zip(all_losses, all_norms, batch_hashes))])
    samples_path.write_text(
        "\n\n".join(f"=== step {s} ===\n{t}" for s, t in samples), encoding="utf-8")
    if scheduled:
        with lr_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "lr"])
            writer.writerows([[s, repr(lr)] for s, lr in step_lrs])
        provenance["fingerprints"]["per_step_lr_log"] = display_path(lr_path)

    provenance["status"] = "completed"
    provenance["fingerprints"].update({
        "batch_starts_running_sha256": running_batch_hash.hexdigest(),
        "per_step_log": display_path(grad_path),
        "generation_draws": {str(s): d for s, d in generation_draws.items()},
        "final_step": total_steps,
        "final_weights_sha256": sha256_tensors(sorted(model.state_dict().items())),
    })
    manifest_path.write_text(json.dumps(provenance, indent=1), encoding="utf-8")

    median_norm, max_norm, spikes = grad_norm_report(all_norms, first_step=start_step + 1)
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
                  checkpoints_dir, device, results_path, args.results_title)
    print(f"\nResults -> {display_path(results_path)}")
    print(f"Provenance -> {display_path(manifest_path)}")


def write_summary(args, config, wall_clock, curve, final_train, final_val, best_val,
                  median_norm, max_norm, spikes, overall_clip, samples,
                  tokens_per_step, train_tokens, log_path, grad_path, samples_path,
                  checkpoints_dir, device, results_path, title):
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
    # Both lines come from the run's own settings. At default arguments they read exactly
    # as the text this file used to hardcode (the known defect in the 6.2 preregistration).
    schedule = config["lr_schedule"]
    lr_text = (f"lr {args.lr}" if schedule["name"] == "constant" else
               f"lr {args.lr} peak (linear warm-up over steps 1-{schedule['warmup_steps']:,}, "
               f"then cosine decay to {schedule['final_lr']} at step {schedule['decay_end_step']:,})")

    lines = [
        f"# {title}",
        "",
        f"## {args.run_name} — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- model: {config['context_length']}-token context, d_model {config['d_model']}, "
        f"{config['n_heads']} heads, d_ff {config['d_ff']}, {config['num_blocks']} Pre-LN blocks, "
        f"tied LM head — {config['parameters']:,} parameters",
        f"- config: {config['new_steps']:,} steps this run (through step "
        f"{total_steps:,}){resumed}, batch {args.batch_size}, {lr_text}, AdamW "
        f"(weight decay {config['weight_decay']} on projection/FFN weights only), "
        f"grad clip {config['grad_clip']}, seed {config['seed']}, device `{device.type}`",
        f"- data: `{config['train_tokens']}` — {train_tokens:,} train tokens; validation is the "
        f"Stage 3 monitoring split (`tinystories_valid.txt` untouched)",
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
