"""
Stage 5.5: structured error analysis of the 9.75k checkpoint.

Read-only. No training.

Configuration (an evaluation setting, NOT a claim about the best strategy —
that question is still open):
    checkpoint  transformer_10k_best.pt (step 9,750)
    decoding    top-k 40, temperature 1.0
    prompts     the frozen 8 in eval/prompts.json
    samples     5 per prompt, seeds 1337-1341 -> 40 generations
    cap         150 new tokens

===========================================================================
SCORING RULES — fixed before any generation was produced. Not to be adjusted
after seeing results.
===========================================================================

CATEGORY 1 — ENTITY / NAME CONSISTENCY (mechanical)

  A "proper noun" is a word matching [A-Z][a-z]+ that is not in NOT_NAMES
  (sentence starters, pronouns, roles like "Mom", dialogue openers like
  "Thank"). This is exactly eval_generation.py's detector.

  1a. First-name fate. Take the first proper noun in prompt+generation.
      - "recurs"     : it appears again in the generation (if the name came
                       from the prompt, one appearance counts; if the
                       generation introduced it, it must appear >= 2 times)
      - "changes"    : it does not recur, but some other proper noun appears
      - "disappears" : it does not recur and no other proper noun appears
      - "no name"    : no proper noun anywhere in prompt+generation

  1b. Uninitiated new name. A proper noun other than the first one counts as
      UNINITIATED unless one of these cues appears in the three words
      immediately before its first mention:
          named, called, met, meet, meets, saw, sees, found, finds
      Scored yes/no per generation: did at least one uninitiated new name
      appear? This is an approximation of "referenced as if already known",
      and it is deliberately crude so it cannot be bent per generation.

CATEGORY 2 — CAUSAL / STORY CONSISTENCY (read and scored against a checklist)

  Four independent yes/no questions per generation. Each is scored by reading
  the text and applying the definition below; no free-form commentary, and
  each "yes" records the specific phrase that triggered it.

  2a. unexplained_object
      A concrete object is referred to with a definite article or possessive
      ("the box", "his kite") when that object has not been mentioned earlier
      in the prompt or generation. Generic scenery tied to a location already
      named (e.g. "the grass" after "the park") does NOT count.

  2b. uncaused_action
      At least one stated action or emotional state cannot be traced to
      anything in the preceding sentences — e.g. a character becomes sad with
      no preceding event, or performs an action whose precondition never
      occurred.

  2c. character_discontinuity
      A character enters after the opening with no introduction, OR a
      character established in the scene stops being referenced while the
      scene continues, with no exit stated.

  2d. contradicted_ending
      The final sentence asserts a state inconsistent with something stated
      earlier (e.g. everyone is happy though the stated problem was never
      resolved; an object described as lost or broken is used intact).

CATEGORY 3 — COMPLETION QUALITY (mechanical)

  3a. reached_eos      : the model emitted <EOS> before the 150-token cap.
  3b. grammatical      : ends on terminal punctuation (. ! ? optionally
                         followed by a closing quote) AND contains no invented
                         words. An invented word is a lowercase alphabetic
                         token of length >= 2 that never appears anywhere in
                         data/tinystories_train_subset.txt (checked with the
                         Stage 2 word-level tokenizer). Capitalised words are
                         exempt, since a novel character name is legitimate.
  3c. repetition       : share of 4-gram occurrences whose 4-gram appears more
                         than once — the same metric as the greedy baseline.

===========================================================================

    venv/bin/python eval_error_analysis.py

Pass 1 writes eval/generations_topk40.json (generations + category 1 and 3
scores). The category 2 checklist is then filled in eval/causal_scores.json
by reading each generation, and re-running merges it into the report.
"""

import json
from datetime import datetime

import torch

from eval_generation import (MAX_NEW_TOKENS, PROMPTS_FILE, ends_cleanly,
                             four_gram_repetition, load_model, name_tracking,
                             proper_nouns)
from eval_sampling import BASE_SEED, sample_generate
from tokenizer.bpe_tokenizer import SAVE_FILE, BPETokenizer
from tokenizer.word_tokenizer import TRAIN_FILE, split_into_tokens
from train_transformer import PROJECT_ROOT, pick_device

CHECKPOINT = ("9.75k", "checkpoints/transformer_10k_best.pt")
TOP_K = 40
TEMPERATURE = 1.0
NUM_SAMPLES = 5
SEEDS = [BASE_SEED + i for i in range(NUM_SAMPLES)]          # 1337-1341
GENERATIONS_FILE = PROJECT_ROOT / "eval" / "generations_topk40.json"
CAUSAL_SCORES_FILE = PROJECT_ROOT / "eval" / "causal_scores.json"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "stage5_error_analysis.md"

INTRO_CUES = {"named", "called", "met", "meet", "meets", "saw", "sees", "found", "finds"}
CAUSAL_KEYS = ["unexplained_object", "uncaused_action",
               "character_discontinuity", "contradicted_ending"]


def corpus_vocabulary():
    """Every word form that occurs in the training corpus (rule 3b)."""
    tokens = split_into_tokens(TRAIN_FILE.read_text(encoding="utf-8"))
    return {t.lower() for t in tokens}


def invented_words(text, vocabulary):
    """Lowercase alphabetic tokens of length >= 2 that the corpus never uses."""
    out = []
    for token in split_into_tokens(text):
        if token.isalpha() and token[0].islower() and len(token) >= 2:
            if token.lower() not in vocabulary:
                out.append(token)
    return out


def uninitiated_names(prompt, generation):
    """Rule 1b: new proper nouns whose first mention has no introduction cue."""
    first_names = proper_nouns(prompt) or proper_nouns(generation)
    first = first_names[0] if first_names else None
    words = generation.replace('"', " ").split()
    flagged, seen = [], set()
    for i, word in enumerate(words):
        clean = word.strip('.,!?"“”\'')
        if clean == first or clean in seen or clean not in proper_nouns(clean):
            continue
        seen.add(clean)
        window = {w.strip('.,!?"“”\'').lower() for w in words[max(0, i - 3):i]}
        if not (window & INTRO_CUES):
            flagged.append(clean)
    return flagged


def score_mechanical(prompt_text, text, token_ids, hit_eos, vocabulary):
    verdict, first_name, gen_names = name_tracking(prompt_text, text)
    invented = invented_words(text, vocabulary)
    clean_end = ends_cleanly(text)
    flagged = uninitiated_names(prompt_text, text)
    return {
        "name_verdict": verdict,
        "first_name": first_name,
        "names_in_generation": gen_names,
        "uninitiated_names": flagged,
        "reached_eos": hit_eos,
        "ends_cleanly": clean_end,
        "invented_words": invented,
        "grammatical": clean_end and not invented,
        "repetition": four_gram_repetition(token_ids),
        "tokens": len(token_ids),
    }


def generate_all():
    device = pick_device()
    tokenizer = BPETokenizer.load(SAVE_FILE)
    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))["prompts"]
    vocabulary = corpus_vocabulary()
    model, step = load_model(CHECKPOINT[1], device)
    print(f"{CHECKPOINT[0]} (step {step:,}) | top-k {TOP_K}, temperature {TEMPERATURE} | "
          f"{len(prompts)} prompts x {NUM_SAMPLES} seeds = {len(prompts) * NUM_SAMPLES} "
          f"generations\n")

    records = []
    for prompt in prompts:
        for seed in SEEDS:
            ids, hit_eos = sample_generate(model, tokenizer, prompt["text"], TEMPERATURE,
                                           TOP_K, None, device, seed)
            text = tokenizer.decode(ids)
            record = {"id": f"{prompt['id']}_{seed}", "prompt_id": prompt["id"],
                      "prompt": prompt["text"], "seed": seed, "text": text}
            record.update(score_mechanical(prompt["text"], text, ids, hit_eos, vocabulary))
            records.append(record)
        done = [r for r in records if r["prompt_id"] == prompt["id"]]
        print(f"  {prompt['id']:<15} eos {sum(r['reached_eos'] for r in done)}/{NUM_SAMPLES} "
              f"| gramm {sum(r['grammatical'] for r in done)}/{NUM_SAMPLES} "
              f"| rep {sum(r['repetition'] for r in done) / NUM_SAMPLES:5.1f}%")

    GENERATIONS_FILE.parent.mkdir(exist_ok=True)
    GENERATIONS_FILE.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"\nWrote {GENERATIONS_FILE.relative_to(PROJECT_ROOT)} ({len(records)} records)")
    return records


def build_report(records, causal):
    n = len(records)
    pct = lambda count: f"{count} / {n} ({100 * count / n:.0f}%)"

    verdicts = {key: sum(1 for r in records if r["name_verdict"] == key)
                for key in ["recurs", "changes", "disappears", "no name"]}
    uninitiated = sum(1 for r in records if r["uninitiated_names"])
    eos = sum(1 for r in records if r["reached_eos"])
    grammatical = sum(1 for r in records if r["grammatical"])
    clean_end = sum(1 for r in records if r["ends_cleanly"])
    with_invented = sum(1 for r in records if r["invented_words"])
    mean_rep = sum(r["repetition"] for r in records) / n

    lines = [
        "# Stage 5.5 — structured error analysis",
        "",
        f"_{datetime.now():%Y-%m-%d %H:%M}_ · read-only, no training.",
        "",
        f"- checkpoint **{CHECKPOINT[0]}** (`{CHECKPOINT[1]}`, step 9,750)",
        f"- decoding **top-k {TOP_K}, temperature {TEMPERATURE}** — an evaluation "
        "configuration for this analysis, not a claim that it is the best strategy",
        f"- {len(records) // NUM_SAMPLES} prompts x {NUM_SAMPLES} seeds "
        f"({SEEDS[0]}-{SEEDS[-1]}) = **{n} generations**, cap {MAX_NEW_TOKENS} tokens",
        "- scoring rules were written and committed in `eval_error_analysis.py` "
        "**before** any generation was produced, and were not adjusted afterwards",
        "",
        "## Category 1 — entity / name consistency (mechanical)",
        "",
        "| outcome | count |",
        "|---|---|",
        f"| first name recurs | {pct(verdicts['recurs'])} |",
        f"| first name changes to another | {pct(verdicts['changes'])} |",
        f"| first name disappears | {pct(verdicts['disappears'])} |",
        f"| no proper noun at all | {pct(verdicts['no name'])} |",
        f"| **at least one uninitiated new name** | **{pct(uninitiated)}** |",
        "",
        "## Category 3 — completion quality (mechanical)",
        "",
        "| measure | count |",
        "|---|---|",
        f"| reached `<EOS>` before the cap | {pct(eos)} |",
        f"| ends on terminal punctuation | {pct(clean_end)} |",
        f"| contains invented words | {pct(with_invented)} |",
        f"| **grammatical (clean end AND no invented words)** | **{pct(grammatical)}** |",
        f"| mean 4-gram repetition | {mean_rep:.1f}% |",
        "",
    ]

    if causal:
        counts = {key: sum(1 for r in records if causal[r["id"]][key]) for key in CAUSAL_KEYS}
        any_issue = sum(1 for r in records if any(causal[r["id"]][k] for k in CAUSAL_KEYS))
        per_gen = sum(sum(causal[r["id"]][k] for k in CAUSAL_KEYS) for r in records) / n
        lines += [
            "## Category 2 — causal / story consistency (checklist, applied by reading)",
            "",
            "| checklist item | generations flagged |",
            "|---|---|",
            f"| unexplained object appears | {pct(counts['unexplained_object'])} |",
            f"| action with no sensible cause | {pct(counts['uncaused_action'])} |",
            f"| character appears/vanishes without transition "
            f"| {pct(counts['character_discontinuity'])} |",
            f"| ending contradicts something earlier | {pct(counts['contradicted_ending'])} |",
            f"| **at least one of the four** | **{pct(any_issue)}** |",
            f"| mean issues per generation | {per_gen:.2f} of 4 |",
            "",
        ]
    else:
        lines += ["## Category 2 — causal / story consistency", "",
                  "_Pending: scores not yet recorded in `eval/causal_scores.json`._", ""]

    lines += ["## Every generation with its scores", ""]
    for record in records:
        flags = []
        if causal:
            scores = causal[record["id"]]
            flags = [key for key in CAUSAL_KEYS if scores[key]]
        lines += [
            f"### `{record['id']}`",
            "",
            f"**Prompt:** `{record['prompt']}`",
            "",
            "```",
            record["prompt"] + record["text"],
            "```",
            "",
            f"- **cat 1**: first name `{record['first_name']}` -> "
            f"**{record['name_verdict']}**; uninitiated new names: "
            f"{record['uninitiated_names'] or 'none'}",
        ]
        if causal:
            evidence = causal[record["id"]].get("evidence", {})
            detail = "; ".join(f"{key} (\"{evidence[key]}\")" for key in flags
                               if key in evidence) or ", ".join(flags)
            lines.append(f"- **cat 2**: {len(flags)}/4 flagged"
                         + (f" — {detail}" if flags else " — none"))
        lines += [
            f"- **cat 3**: {record['tokens']} tokens, "
            f"{'reached <EOS>' if record['reached_eos'] else 'hit the cap'}; "
            f"ends cleanly: {record['ends_cleanly']}; invented words: "
            f"{record['invented_words'] or 'none'}; grammatical: "
            f"**{record['grammatical']}**; repetition {record['repetition']:.1f}%",
            "",
        ]

    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


def main():
    if GENERATIONS_FILE.exists():
        records = json.loads(GENERATIONS_FILE.read_text(encoding="utf-8"))
        print(f"Reusing {GENERATIONS_FILE.relative_to(PROJECT_ROOT)} "
              f"({len(records)} generations)")
    else:
        records = generate_all()

    causal = None
    if CAUSAL_SCORES_FILE.exists():
        causal = json.loads(CAUSAL_SCORES_FILE.read_text(encoding="utf-8"))["scores"]
        missing = [r["id"] for r in records if r["id"] not in causal]
        assert not missing, f"causal scores missing for: {missing}"
    build_report(records, causal)


if __name__ == "__main__":
    main()
