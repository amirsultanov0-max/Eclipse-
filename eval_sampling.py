"""
Stage 5b: sampling strategies vs the greedy baseline.

Read-only. No training. Same frozen prompts as eval_generation.py, so every
number here is comparable to the greedy run.

Strategies tested (all with the same 150-token cap):
    greedy            argmax, the Stage 5a baseline
    temperature 0.7   sharpened distribution
    temperature 1.0   the model's distribution, untouched
    top-k 40          sample among the 40 most likely tokens
    top-p 0.9         sample from the smallest set covering 90% of the mass

Scope (deliberately bounded — not a full checkpoint x strategy grid):
    - 9.75k checkpoint on all 8 prompts
    - 5k checkpoint on the two prompts that looped worst under greedy
      decoding (tom_ball, lily_key)

Seeds: torch.manual_seed(BASE_SEED + sample_index) before each generation, so
sample 1/2/3 always use 1337/1338/1339 whatever the strategy. The same seed
across strategies makes this a paired comparison, and reruns reproduce exactly.

    venv/bin/python eval_sampling.py
"""

import json
from datetime import datetime

import torch
import torch.nn.functional as F

from eval_generation import (MAX_NEW_TOKENS, PROMPTS_FILE, four_gram_repetition,
                             greedy_generate, load_model)
from model.transformer import CONTEXT_LENGTH
from tokenizer.bpe_tokenizer import SAVE_FILE, BPETokenizer
from train_transformer import PROJECT_ROOT, pick_device

OUTPUT_FILE = PROJECT_ROOT / "eval" / "stage5_sampling_eval.md"
BASE_SEED = 1337
SAMPLES_PER_COMBO = 3
LOOP_PROMPTS = ["tom_ball", "lily_key"]

CHECKPOINT_5K = ("5k", "checkpoints/transformer_best.pt")
CHECKPOINT_10K = ("9.75k", "checkpoints/transformer_10k_best.pt")

# (label, temperature, top_k, top_p)
STRATEGIES = [
    ("temp 0.7", 0.7, None, None),
    ("temp 1.0", 1.0, None, None),
    ("top-k 40", 1.0, 40, None),
    ("top-p 0.9", 1.0, None, 0.9),
]


def top_k_filter(logits, k):
    """Keep the k highest logits, make the rest impossible."""
    kth_value = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def top_p_filter(logits, p):
    """
    Keep the smallest set of tokens whose probabilities first reach p.

    Unlike top-k, the size of that set changes with the model's confidence: a
    confident step may keep 2 tokens, an uncertain one several hundred.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # `cumulative - probs` is the mass BEFORE this token, so the first token to
    # cross p is still kept and everything after it is dropped.
    sorted_logits[cumulative - probs > p] = float("-inf")
    restored = torch.full_like(logits, float("-inf"))
    restored[sorted_indices] = sorted_logits
    return restored


@torch.no_grad()
def sample_generate(model, tokenizer, prompt, temperature, top_k, top_p, device, seed):
    torch.manual_seed(seed)
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated, hit_eos = [], False
    for _ in range(MAX_NEW_TOKENS):
        logits = model(idx[:, -CONTEXT_LENGTH:])[0][0, -1] / temperature
        if top_k is not None:
            logits = top_k_filter(logits, top_k)
        if top_p is not None:
            logits = top_p_filter(logits, top_p)
        next_id = int(torch.multinomial(F.softmax(logits, dim=-1), num_samples=1))
        if next_id == tokenizer.eos_id:
            hit_eos = True
            break
        generated.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return generated, hit_eos


def run_combo(model, tokenizer, prompt_text, strategy, device):
    """One (prompt, strategy) cell: SAMPLES_PER_COMBO generations at fixed seeds."""
    label, temperature, top_k, top_p = strategy
    runs = []
    for i in range(SAMPLES_PER_COMBO):
        seed = BASE_SEED + i
        ids, hit_eos = sample_generate(model, tokenizer, prompt_text, temperature,
                                       top_k, top_p, device, seed)
        runs.append({"seed": seed, "text": tokenizer.decode(ids), "tokens": len(ids),
                     "hit_eos": hit_eos, "repetition": four_gram_repetition(ids)})
    return runs


def evaluate(checkpoint, prompts, tokenizer, device):
    """Returns {prompt_id: {strategy_label: [runs]}} including a greedy row."""
    label, path = checkpoint
    model, step = load_model(path, device)
    print(f"\n{label} ({path}, step {step:,})")
    table = {}
    for prompt in prompts:
        ids, hit_eos = greedy_generate(model, tokenizer, prompt["text"],
                                       MAX_NEW_TOKENS, device)
        cells = {"greedy": [{"seed": None, "text": tokenizer.decode(ids),
                             "tokens": len(ids), "hit_eos": hit_eos,
                             "repetition": four_gram_repetition(ids)}]}
        for strategy in STRATEGIES:
            cells[strategy[0]] = run_combo(model, tokenizer, prompt["text"],
                                           strategy, device)
        table[prompt["id"]] = cells
        summary = "  ".join(
            f"{name} {sum(r['repetition'] for r in runs) / len(runs):5.1f}%"
            for name, runs in cells.items())
        print(f"  {prompt['id']:<15} {summary}")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"label": label, "path": path, "step": step, "table": table}


def mean_repetition(runs):
    return sum(r["repetition"] for r in runs) / len(runs)


def write_report(run_10k, run_5k, prompts):
    strategy_names = ["greedy"] + [s[0] for s in STRATEGIES]
    prompt_by_id = {p["id"]: p for p in prompts}

    lines = [
        "# Stage 5b — sampling strategies vs greedy",
        "",
        f"_{datetime.now():%Y-%m-%d %H:%M}_ · read-only evaluation, no training.",
        "",
        f"- prompts: the frozen set in `eval/prompts.json` (same 8 as the greedy baseline)",
        f"- cap {MAX_NEW_TOKENS} new tokens, identical for every strategy",
        f"- **seeds: `torch.manual_seed({BASE_SEED} + sample_index)`** before each "
        f"generation, so samples 1-{SAMPLES_PER_COMBO} always use "
        f"{', '.join(str(BASE_SEED + i) for i in range(SAMPLES_PER_COMBO))} — the same "
        "seeds across every strategy and prompt, and reruns reproduce exactly",
        f"- {SAMPLES_PER_COMBO} samples per (prompt, strategy); greedy is deterministic so "
        "it has one",
        f"- scope: **9.75k** on all 8 prompts; **5k** on the two prompts that looped worst "
        f"under greedy ({', '.join(LOOP_PROMPTS)})",
        "",
        "Repetition = share of 4-gram occurrences in the generation whose 4-gram appears "
        "more than once, the same metric as the greedy baseline.",
        "",
        "## 9.75k across all 8 prompts — mean 4-gram repetition",
        "",
        "| prompt | " + " | ".join(strategy_names) + " |",
        "|---" * (len(strategy_names) + 1) + "|",
    ]
    for prompt in prompts:
        cells = run_10k["table"][prompt["id"]]
        row = " | ".join(f"{mean_repetition(cells[name]):.1f}%" for name in strategy_names)
        lines.append(f"| {prompt['id']} | {row} |")

    overall = {name: sum(mean_repetition(run_10k["table"][p["id"]][name]) for p in prompts)
               / len(prompts) for name in strategy_names}
    lines.append("| **mean** | " + " | ".join(f"**{overall[n]:.1f}%**"
                                              for n in strategy_names) + " |")

    lengths = {name: sum(sum(r["tokens"] for r in run_10k["table"][p["id"]][name])
                         / len(run_10k["table"][p["id"]][name]) for p in prompts) / len(prompts)
               for name in strategy_names}
    eos = {name: sum(sum(1 for r in run_10k["table"][p["id"]][name] if r["hit_eos"])
                     for p in prompts) for name in strategy_names}
    totals = {name: sum(len(run_10k["table"][p["id"]][name]) for p in prompts)
              for name in strategy_names}
    lines += [
        "| mean length | " + " | ".join(f"{lengths[n]:.0f}" for n in strategy_names) + " |",
        "| stopped at `<EOS>` | " + " | ".join(f"{eos[n]}/{totals[n]}"
                                               for n in strategy_names) + " |",
        "",
        "## The two looping prompts, per sample",
        "",
        "Each cell is one generation's repetition rate, at the seed shown.",
        "",
        "| prompt | ckpt | strategy | " + " | ".join(
            f"seed {BASE_SEED + i}" for i in range(SAMPLES_PER_COMBO)) + " | mean |",
        "|---" * (4 + SAMPLES_PER_COMBO) + "|",
    ]
    for prompt_id in LOOP_PROMPTS:
        for run in (run_5k, run_10k):
            for name in strategy_names:
                runs = run["table"][prompt_id][name]
                cells = " | ".join(f"{r['repetition']:.1f}%" for r in runs)
                padding = " | —" * (SAMPLES_PER_COMBO - len(runs))
                lines.append(f"| {prompt_id} | {run['label']} | {name} | {cells}{padding} "
                             f"| {mean_repetition(runs):.1f}% |")

    lines += ["", "## Full generations", ""]
    for run in (run_10k, run_5k):
        lines += [f"## Checkpoint {run['label']} (`{run['path']}`, step {run['step']:,})", ""]
        for prompt_id, cells in run["table"].items():
            prompt = prompt_by_id[prompt_id]
            lines += [f"### `{prompt_id}` — prompt: `{prompt['text']}`", ""]
            for name in strategy_names:
                for i, r in enumerate(cells[name], start=1):
                    seed = f"seed {r['seed']}" if r["seed"] is not None else "deterministic"
                    ending = "stopped at <EOS>" if r["hit_eos"] else "hit the cap"
                    lines += [
                        f"**{name}**, sample {i} ({seed}, {r['tokens']} tokens, {ending}, "
                        f"repetition {r['repetition']:.1f}%):",
                        "", "```", prompt["text"] + r["text"], "```", "",
                    ]
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    device = pick_device()
    tokenizer = BPETokenizer.load(SAVE_FILE)
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))["prompts"]
    loop_prompts = [p for p in prompts if p["id"] in LOOP_PROMPTS]
    print(f"{len(prompts)} prompts | {len(STRATEGIES)} sampling strategies + greedy | "
          f"{SAMPLES_PER_COMBO} samples each | seeds {BASE_SEED}-"
          f"{BASE_SEED + SAMPLES_PER_COMBO - 1} | device {device.type}")

    run_10k = evaluate(CHECKPOINT_10K, prompts, tokenizer, device)
    run_5k = evaluate(CHECKPOINT_5K, loop_prompts, tokenizer, device)

    write_report(run_10k, run_5k, prompts)
    print(f"\nWrote {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
