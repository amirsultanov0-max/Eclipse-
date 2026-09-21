"""
Stage 5: greedy-decoding comparison between the 5k and 9.75k checkpoints.

Read-only. No training, no sampling — greedy decoding (argmax at every step)
makes each generation deterministic, so the only variable is the checkpoint.

Five metrics, all computed mechanically, no human judgement:

  1. 4-gram repetition rate — of all 4-grams in the generation (BPE tokens),
     the share that occur more than once. 0% means nothing repeats.
  2. Length before <EOS> — generated tokens until the model chose <EOS>, or the
     cap if it never did.
  3. Name tracking — the first proper noun in prompt+generation, then whether
     it recurs in the generation, is replaced by a different name ("changes"),
     or is never mentioned again ("disappears").
  4. Quotation balance — straight quotes must be even and curly quotes must
     pair up.
  5. Ending — does the text stop on terminal punctuation (optionally followed
     by a closing quote), or mid-sentence?

    venv/bin/python eval_generation.py
"""

import json
import re
from collections import Counter
from datetime import datetime

import torch

from model.transformer import CONTEXT_LENGTH, TinyTransformer
from tokenizer.bpe_tokenizer import SAVE_FILE, BPETokenizer
from train_transformer import PROJECT_ROOT, pick_device

PROMPTS_FILE = PROJECT_ROOT / "eval" / "prompts.json"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "stage5_generation_eval.md"
MAX_NEW_TOKENS = 150
CHECKPOINTS = [
    ("5k", "checkpoints/transformer_best.pt"),
    ("9.75k", "checkpoints/transformer_10k_best.pt"),
]

# Capitalised words that are ordinary sentence starters or roles rather than
# character names. Without this list every sentence-initial "The" would count
# as a proper noun.
NOT_NAMES = {
    "The", "A", "An", "And", "But", "So", "Then", "When", "While", "After", "As",
    "One", "Once", "Every", "Now", "Later", "Finally", "Suddenly", "Soon", "Today",
    "He", "She", "It", "They", "We", "You", "I", "His", "Her", "Their", "My", "Our",
    "This", "That", "There", "These", "Those", "If", "Because", "From", "In", "On",
    "At", "To", "With", "For", "Mom", "Mommy", "Dad", "Daddy", "Mum", "Yes", "No",
    "Oh", "Ok", "Okay", "Let", "Do", "Don", "What", "Why", "How", "Where", "Who",
    # Dialogue openers and imperatives: TinyStories is full of quoted speech, so
    # these turn up capitalised mid-sentence without being anyone's name.
    "Thank", "Thanks", "Hi", "Hello", "Hey", "Wow", "Please", "Sorry", "Look",
    "Come", "Go", "Stop", "Wait", "Help", "See", "Here", "Good", "Great", "Sure",
    "Can", "Will", "Would", "Could", "Should", "Did", "Was", "Is", "Are", "Have",
    "Well", "Maybe", "Just", "Never", "Always", "Something", "Someone", "Everyone",
}
WORD_RE = re.compile(r"\b[A-Z][a-z]+\b")


def proper_nouns(text):
    return [w for w in WORD_RE.findall(text) if w not in NOT_NAMES]


def four_gram_repetition(token_ids):
    """Share of 4-gram occurrences whose 4-gram appears more than once."""
    if len(token_ids) < 4:
        return 0.0
    grams = [tuple(token_ids[i:i + 4]) for i in range(len(token_ids) - 3)]
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return 100 * repeated / len(grams)


def name_tracking(prompt, generation):
    """Returns (verdict, first_name, names_in_generation)."""
    prompt_names = proper_nouns(prompt)
    gen_names = proper_nouns(generation)

    if prompt_names:
        first = prompt_names[0]
        # The name already exists, so one mention in the generation is a recurrence.
        recurs = gen_names.count(first) >= 1
    elif gen_names:
        first = gen_names[0]
        # The generation introduced it, so it must appear at least twice.
        recurs = gen_names.count(first) >= 2
    else:
        return "no name", None, []

    if recurs:
        return "recurs", first, gen_names
    if [n for n in gen_names if n != first]:
        return "changes", first, gen_names
    return "disappears", first, gen_names


def quote_balance(text):
    straight = text.count('"')
    curly_open, curly_close = text.count("“"), text.count("”")
    balanced = straight % 2 == 0 and curly_open == curly_close
    return balanced, straight + curly_open + curly_close


def ends_cleanly(text):
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in ".!?":
        return True
    # A closing quote right after terminal punctuation still counts as finished.
    return stripped[-1] in '"”' and len(stripped) > 1 and stripped[-2] in ".!?"


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_new_tokens, device):
    """Argmax at every step: deterministic, no sampling, no temperature."""
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated, hit_eos = [], False
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -CONTEXT_LENGTH:])
        next_id = int(logits[0, -1].argmax())
        if next_id == tokenizer.eos_id:
            hit_eos = True
            break
        generated.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return generated, hit_eos


def load_model(relative_path, device):
    checkpoint = torch.load(PROJECT_ROOT / relative_path, map_location=device,
                            weights_only=False)
    model = TinyTransformer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["step"]


def evaluate_checkpoint(label, relative_path, prompts, tokenizer, device):
    model, step = load_model(relative_path, device)
    print(f"{label}: {relative_path} (step {step:,})")
    results = []
    for prompt in prompts:
        token_ids, hit_eos = greedy_generate(model, tokenizer, prompt["text"],
                                             MAX_NEW_TOKENS, device)
        text = tokenizer.decode(token_ids)
        verdict, first_name, gen_names = name_tracking(prompt["text"], text)
        balanced, quote_count = quote_balance(text)
        results.append({
            "prompt": prompt, "text": text, "tokens": len(token_ids),
            "hit_eos": hit_eos, "repetition": four_gram_repetition(token_ids),
            "name_verdict": verdict, "first_name": first_name, "names": gen_names,
            "quotes_balanced": balanced, "quote_count": quote_count,
            "ends_cleanly": ends_cleanly(text),
        })
        print(f"  {prompt['id']:<15} {len(token_ids):>4} tokens | "
              f"rep {four_gram_repetition(token_ids):>5.1f}% | {verdict:<11} | "
              f"eos {str(hit_eos):<5} | clean end {results[-1]['ends_cleanly']}")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"label": label, "path": relative_path, "step": step, "results": results}


def summarise(run):
    results = run["results"]
    n = len(results)
    verdicts = Counter(r["name_verdict"] for r in results)
    return {
        "repetition": sum(r["repetition"] for r in results) / n,
        "tokens": sum(r["tokens"] for r in results) / n,
        "eos_count": sum(1 for r in results if r["hit_eos"]),
        "verdicts": verdicts,
        "quotes_ok": sum(1 for r in results if r["quotes_balanced"]),
        "clean_ends": sum(1 for r in results if r["ends_cleanly"]),
        "n": n,
    }


def write_report(runs, prompts):
    a, b = runs
    sa, sb = summarise(a), summarise(b)
    n = sa["n"]

    def verdict_cell(summary):
        order = ["recurs", "changes", "disappears", "no name"]
        return ", ".join(f"{k} {summary['verdicts'][k]}" for k in order
                         if summary["verdicts"][k])

    lines = [
        "# Stage 5 — greedy generation comparison",
        "",
        f"_{datetime.now():%Y-%m-%d %H:%M}_ · read-only evaluation, no training.",
        "",
        f"- checkpoints: **{a['label']}** (`{a['path']}`, step {a['step']:,}) vs "
        f"**{b['label']}** (`{b['path']}`, step {b['step']:,})",
        f"- {n} fixed prompts from `eval/prompts.json`, identical for both checkpoints",
        f"- greedy decoding (argmax, no sampling), cap {MAX_NEW_TOKENS} new tokens",
        "- deterministic: rerunning this script reproduces every generation exactly",
        "",
        "## Metrics, averaged across all prompts",
        "",
        f"| metric | {a['label']} | {b['label']} |",
        "|---|---|---|",
        f"| 4-gram repetition rate | {sa['repetition']:.1f}% | {sb['repetition']:.1f}% |",
        f"| generation length (tokens) | {sa['tokens']:.1f} | {sb['tokens']:.1f} |",
        f"| stopped at `<EOS>` | {sa['eos_count']}/{n} | {sb['eos_count']}/{n} |",
        f"| name tracking | {verdict_cell(sa)} | {verdict_cell(sb)} |",
        f"| quotation marks balanced | {sa['quotes_ok']}/{n} | {sb['quotes_ok']}/{n} |",
        f"| ends on terminal punctuation | {sa['clean_ends']}/{n} | {sb['clean_ends']}/{n} |",
        "",
        "### How each metric is computed",
        "",
        "- **4-gram repetition rate**: of all 4-grams in the generation (BPE tokens), the "
        "share of occurrences whose 4-gram appears more than once. 0% = nothing repeats.",
        "- **Generation length**: tokens produced before the model chose `<EOS>`; equal to "
        f"the {MAX_NEW_TOKENS}-token cap when it never did.",
        "- **Name tracking**: the first proper noun in prompt+generation (capitalised word, "
        "excluding sentence starters, pronouns and roles like \"Mom\"). *recurs* = it appears "
        "again in the generation, *changes* = it does not but another name does, "
        "*disappears* = no name after the first mention.",
        "- **Quotation balance**: straight quotes even and curly quotes paired.",
        "- **Ending**: last character is `.`/`!`/`?`, optionally followed by a closing quote.",
        "",
        "## Per-prompt metrics",
        "",
        f"| prompt | ckpt | tokens | `<EOS>` | rep. | name | quotes | clean end |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, prompt in enumerate(prompts):
        for run in runs:
            r = run["results"][i]
            name = f"{r['first_name']} ({r['name_verdict']})" if r["first_name"] else "—"
            quotes = "ok" if r["quotes_balanced"] else f"UNBALANCED ({r['quote_count']})"
            eos = "yes" if r["hit_eos"] else "no"
            clean = "yes" if r["ends_cleanly"] else "no"
            lines.append(
                f"| {prompt['id']} | {run['label']} | {r['tokens']} | {eos} "
                f"| {r['repetition']:.1f}% | {name} | {quotes} | {clean} |")

    lines += ["", "## Full generations", ""]
    for i, prompt in enumerate(prompts):
        lines += [f"### {i + 1}. `{prompt['id']}` — {prompt['probes']}", "",
                  f"**Prompt:** `{prompt['text']}`", ""]
        for run in runs:
            r = run["results"][i]
            lines += [
                f"**{run['label']}** ({r['tokens']} tokens, "
                f"{'stopped at <EOS>' if r['hit_eos'] else 'hit the length cap'}, "
                f"repetition {r['repetition']:.1f}%):",
                "",
                "```",
                prompt["text"] + r["text"],
                "```",
                "",
            ]
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    device = pick_device()
    tokenizer = BPETokenizer.load(SAVE_FILE)
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))["prompts"]
    print(f"{len(prompts)} prompts | greedy decoding | cap {MAX_NEW_TOKENS} tokens | "
          f"device {device.type}\n")

    runs = [evaluate_checkpoint(label, path, prompts, tokenizer, device)
            for label, path in CHECKPOINTS]

    write_report(runs, prompts)
    a, b = (summarise(run) for run in runs)
    print(f"\n{'metric':<30} {CHECKPOINTS[0][0]:>10} {CHECKPOINTS[1][0]:>10}")
    print(f"{'4-gram repetition':<30} {a['repetition']:>9.1f}% {b['repetition']:>9.1f}%")
    print(f"{'mean length (tokens)':<30} {a['tokens']:>10.1f} {b['tokens']:>10.1f}")
    print(f"{'stopped at <EOS>':<30} {a['eos_count']:>7}/{a['n']} {b['eos_count']:>7}/{b['n']}")
    print(f"{'quotes balanced':<30} {a['quotes_ok']:>7}/{a['n']} {b['quotes_ok']:>7}/{b['n']}")
    print(f"{'clean endings':<30} {a['clean_ends']:>7}/{a['n']} {b['clean_ends']:>7}/{b['n']}")
    for key in ["recurs", "changes", "disappears", "no name"]:
        print(f"{'names ' + key:<30} {a['verdicts'][key]:>10} {b['verdicts'][key]:>10}")
    print(f"\nWrote {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
