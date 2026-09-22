"""
Stage 6.2a compute pilot: can this Mac train a wider model, and how long would it take?

Feasibility only, not model quality. For ONE architecture per process it runs a
short training burst on the real workload and measures speed and memory:

    200 steps (the first 20 are warm-up and are not timed), batch 16, T=512,
    AdamW lr 1e-3, weight decay 0.01 on Linear weights, grad clip 1.0,
    on the 200 MB training stream (data/stage6_1_train_tokens.npy).

Everything is imported from train_transformer.py (model, batch sampler, param
groups, monitoring eval), so the only thing that differs from a real run is the
architecture. Each step does the real loop's per-step work: draw a batch,
fingerprint its start positions, forward, backward, clip, AdamW step, and read
loss and grad norm back to the CPU (which also synchronises the GPU, so the
per-step wall-clock times are real, not just queued work).

After the burst it times ONE monitoring-eval pass (the same 50 fixed batches a
real run evaluates every 250 steps), because a 10k-step projection has to
include the 40 evals a real run does. Nothing touches tinystories_valid.txt.

One architecture per process, so the peak RSS is that candidate's alone and no
state carries over between candidates. Writes one JSON file and nothing else:
no checkpoints, no samples, no results markdown.

    python scripts/pilot_6_2a.py --count-only --d-model 224 --n-heads 4
    python scripts/pilot_6_2a.py --name d224_h4 --d-model 224 --n-heads 4
"""

import argparse
import hashlib
import json
import platform
import re
import resource
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.transformer import CONTEXT_LENGTH, NUM_BLOCKS, VOCAB_SIZE, TinyTransformer  # noqa: E402
from train_transformer import (BATCH_SIZE, EVAL_BATCHES, EVAL_EVERY, EVAL_SEED,  # noqa: E402
                               GRAD_CLIP, LEARNING_RATE, SEED, VAL_TOKENS, WEIGHT_DECAY,
                               evaluate, get_batch, git_provenance, load_stream,
                               make_fixed_batches, make_param_groups, sha256_file)

TRAIN_TOKENS = ROOT / "data" / "stage6_1_train_tokens.npy"
TRAIN_SHA256 = "5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb"
OUT_DIR = ROOT / "results" / "pilot_6_2a"
PROJECT_STEPS = 10_000
MONITOR_EVERY_S = 2.0       # how often the background thread samples memory pressure / swap


def architecture(d_model, n_heads, d_ff, blocks):
    """Exact parameter counts, straight from an instantiated model (CPU, no training)."""
    model = TinyTransformer(d_model=d_model, num_heads=n_heads, d_ff=d_ff, num_blocks=blocks)
    total = model.num_parameters()
    # The LM head is tied to the token embedding, so these rows count once.
    embedding = model.token_embedding.weight.numel() + model.position_embedding.weight.numel()
    return model, {"d_model": d_model, "n_heads": n_heads, "head_dim": d_model // n_heads,
                   "d_ff": d_ff, "blocks": blocks, "vocab": VOCAB_SIZE,
                   "context": CONTEXT_LENGTH, "params": total, "embedding_params": embedding,
                   "embedding_share": embedding / total}


# --- system memory monitoring (macOS, read-only) ------------------------------

def read_system_memory():
    """Pressure level (1 normal, 2 warn, 4 critical), swap used (MB), cumulative swapouts."""
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level", "vm.swapusage"],
                         capture_output=True, text=True).stdout.splitlines()
    level = int(out[0])
    swap_used = float(re.search(r"used = ([\d.]+)M", out[1]).group(1))
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    swapouts = int(re.search(r"Swapouts:\s+(\d+)", vm).group(1))
    return {"t": time.time(), "pressure_level": level, "swap_used_mb": swap_used,
            "swapouts": swapouts}


class MemoryMonitor(threading.Thread):
    """Samples system memory every few seconds until stopped."""

    def __init__(self):
        super().__init__(daemon=True)
        # (not self._stop: threading.Thread uses that name internally)
        self.samples, self._halt = [read_system_memory()], threading.Event()

    def run(self):
        while not self._halt.wait(MONITOR_EVERY_S):
            self.samples.append(read_system_memory())

    def stop(self):
        self._halt.set()
        self.join()
        self.samples.append(read_system_memory())
        first, s = self.samples[0], self.samples
        return {"samples": len(s),
                "pressure_level_max": max(x["pressure_level"] for x in s),
                "pressure_levels_seen": sorted({x["pressure_level"] for x in s}),
                "swap_used_mb_start": first["swap_used_mb"],
                "swap_used_mb_max": max(x["swap_used_mb"] for x in s),
                "swapouts_during_run": s[-1]["swapouts"] - first["swapouts"]}


def thermal_note():
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return " | ".join(line.strip() for line in out.splitlines() if line.strip())


def power_source():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return out.splitlines()[0].strip() if out else "unknown"


# --- the measurement ----------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Stage 6.2a compute pilot (one architecture)")
    p.add_argument("--name", help="output name, e.g. d224_h4 (results/pilot_6_2a/<name>.json)")
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--n-heads", type=int, required=True)
    p.add_argument("--d-ff", type=int, default=None, help="default 4 x d_model (the scaling rule)")
    p.add_argument("--blocks", type=int, default=NUM_BLOCKS)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20, help="leading steps excluded from timing")
    p.add_argument("--count-only", action="store_true",
                   help="print the architecture and exact parameter count, then exit (no device work)")
    args = p.parse_args()
    d_ff = args.d_ff or 4 * args.d_model

    model, arch = architecture(args.d_model, args.n_heads, d_ff, args.blocks)
    if args.count_only:
        print(json.dumps(arch, indent=1))
        return

    # --- preconditions: fail in seconds, before any compute ------------------
    if not args.name:
        raise SystemExit("--name is required for a measurement run")
    out_path = OUT_DIR / f"{args.name}.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path.relative_to(ROOT)}")
    git_info, _ = git_provenance()
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                              "scripts/pilot_6_2a.py"], capture_output=True).returncode == 0
    if git_info["dirty"] or not tracked:
        raise SystemExit("tree is dirty or this script is not committed: "
                         "commit first so the result points at the exact code that ran")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available: the pilot is only meaningful on the training device")
    train_sha = sha256_file(TRAIN_TOKENS)
    if train_sha != TRAIN_SHA256:
        raise SystemExit(f"training stream hash mismatch: {train_sha}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now().isoformat(timespec="seconds")
    thermal_before, power = thermal_note(), power_source()
    monitor = MemoryMonitor()
    monitor.start()

    # --- same setup order as train_transformer.main() -----------------------
    torch.manual_seed(SEED)
    device = torch.device("mps")
    train_stream = load_stream(TRAIN_TOKENS, device)
    val_stream = load_stream(VAL_TOKENS, device)
    val_batches = make_fixed_batches(val_stream, EVAL_BATCHES, BATCH_SIZE, seed=EVAL_SEED)
    # Built after the seed and the data, as in the real run, so initial weights
    # come from the same point in the random stream.
    model, _ = architecture(args.d_model, args.n_heads, d_ff, args.blocks)
    model = model.to(device)
    groups, _ = make_param_groups(model, WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)

    step_seconds, losses = [], []
    allocated_peak, driver_peak = 0, 0

    def probe():
        # Allocator counters only: a CPU-side read, no GPU synchronisation.
        nonlocal allocated_peak, driver_peak
        allocated_peak = max(allocated_peak, torch.mps.current_allocated_memory())
        driver_peak = max(driver_peak, torch.mps.driver_allocated_memory())

    print(f"pilot {args.name}: d_model {args.d_model}, heads {args.n_heads}, d_ff {d_ff}, "
          f"{args.blocks} blocks, {arch['params']:,} params | {args.steps} steps "
          f"(first {args.warmup} untimed)")
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        # Identical to the real loop body (get_batch + fingerprint + train_step),
        # written out so memory can be read after the forward and backward passes.
        x, y, starts = get_batch(train_stream, BATCH_SIZE, return_starts=True)
        hashlib.sha256(starts.cpu().numpy().tobytes()).hexdigest()
        _, loss = model(x, y)
        probe()                                  # all activations are held here
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        probe()                                  # gradients now exist
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        probe()                                  # AdamW moments now exist
        losses.append(loss.item())
        grad_norm.item()                         # the real loop reads both back
        step_seconds.append(time.perf_counter() - t0)
        if step % 20 == 0:
            print(f"  step {step:>4} | loss {losses[-1]:.3f} | {1000 * step_seconds[-1]:.0f} ms")

    # One monitoring-eval pass, as a real run does every EVAL_EVERY steps.
    t0 = time.perf_counter()
    evaluate(model, val_batches)
    eval_seconds = time.perf_counter() - t0

    system = monitor.stop()
    timed = step_seconds[args.warmup:]
    mean_s = statistics.mean(timed)
    tokens_per_step = BATCH_SIZE * CONTEXT_LENGTH
    evals_per_run = PROJECT_STEPS // EVAL_EVERY
    train_h = mean_s * PROJECT_STEPS / 3600
    eval_h = eval_seconds * evals_per_run / 3600

    result = {
        "name": args.name, "started_at": started,
        "architecture": arch,
        "workload": {"steps": args.steps, "warmup_steps_discarded": args.warmup,
                     "timed_steps": len(timed), "batch_size": BATCH_SIZE,
                     "tokens_per_step": tokens_per_step, "lr": LEARNING_RATE,
                     "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "seed": SEED,
                     "train_tokens": str(TRAIN_TOKENS.relative_to(ROOT)),
                     "train_tokens_sha256": train_sha, "train_tokens_count": len(train_stream)},
        "timing": {"ms_per_step_mean": 1000 * mean_s,
                   "ms_per_step_median": 1000 * statistics.median(timed),
                   "ms_per_step_min": 1000 * min(timed), "ms_per_step_max": 1000 * max(timed),
                   "ms_per_step_first_10_timed": 1000 * statistics.mean(timed[:10]),
                   "ms_per_step_last_10_timed": 1000 * statistics.mean(timed[-10:]),
                   "tokens_per_sec": tokens_per_step / mean_s,
                   "warmup_ms_per_step": [1000 * s for s in step_seconds[:args.warmup]],
                   "monitoring_eval_seconds": eval_seconds,
                   "all_step_ms": [1000 * s for s in step_seconds]},
        "memory": {"torch_allocated_peak_sampled_mb": allocated_peak / 2**20,
                   "torch_driver_allocated_peak_mb": driver_peak / 2**20,
                   "mps_recommended_max_mb": torch.mps.recommended_max_memory() / 2**20,
                   "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                   "system": system},
        "projection": {"basis": f"mean timed ms/step x {PROJECT_STEPS:,} steps + "
                                f"{evals_per_run} monitoring evals; excludes sample generation, "
                                "checkpoint writes and official evaluation",
                       "train_hours_10k": train_h, "eval_hours_10k": eval_h,
                       "run_hours_10k": train_h + eval_h,
                       "three_seed_hours": 3 * (train_h + eval_h)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": "mps", "power": power,
                        "thermal_before": thermal_before, "thermal_after": thermal_note()},
        "git": git_info,
        "losses": losses,
    }
    out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    r = result
    print(f"\n{args.name}: {r['timing']['ms_per_step_mean']:.0f} ms/step "
          f"({r['timing']['tokens_per_sec']:,.0f} tok/s) | torch peak "
          f"{r['memory']['torch_allocated_peak_sampled_mb']:,.0f} MB sampled, "
          f"driver {r['memory']['torch_driver_allocated_peak_mb']:,.0f} MB | RSS "
          f"{r['memory']['process_peak_rss_mb']:,.0f} MB | pressure max "
          f"{system['pressure_level_max']} | swapouts {system['swapouts_during_run']}")
    print(f"projected 10k run {r['projection']['run_hours_10k']:.2f} h, three seeds "
          f"{r['projection']['three_seed_hours']:.2f} h -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
