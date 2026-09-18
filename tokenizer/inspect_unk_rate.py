"""
Read-only inspection: how well does the word-level vocab (built from the
training subset) cover unseen text? Encodes the first 200 validation stories.
Nothing is trained, saved, or added to the vocabulary.
"""

import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path.home() / "tiny-lm"
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

from word_tokenizer import (  # the existing, unmodified tokenizer
    EOS, STORY_SEPARATOR, TRAIN_FILE, UNK, WordTokenizer, split_into_tokens,
)

VALID_FILE = PROJECT_ROOT / "data" / "tinystories_valid.txt"
NUM_STORIES = 200


def section(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# --- The existing vocabulary (derived from the training file only) ---------
train_counts = Counter(t for t in split_into_tokens(TRAIN_FILE.read_text(encoding="utf-8")) if t != EOS)
tok = WordTokenizer(train_counts)

# Confirm it is the exact vocab from the previous run.
assert len(tok) == 12_996
assert tok.token_to_id["the"] == 5 and tok.token_to_id["needle"] == 2051 and tok.token_to_id["lily"] == 3034

# --- Read only the first 200 validation stories ----------------------------
lines, seen = [], 0
with VALID_FILE.open(encoding="utf-8") as f:
    for line in f:
        lines.append(line)
        if line.strip() == STORY_SEPARATOR:
            seen += 1
            if seen == NUM_STORIES:
                break
sample = "".join(lines)

# --- Encode (lookup only) ---------------------------------------------------
tokens = split_into_tokens(sample)
ids = tok.encode(sample)
num_eos = tokens.count(EOS)
num_real = len(tokens) - num_eos
unk_words = Counter(t for t, i in zip(tokens, ids) if i == tok.unk_id)
num_unk = sum(unk_words.values())

section("SAMPLE")
print(f"Stories read:          {num_eos}  (lines read: {len(lines):,}, chars: {len(sample):,})")
print(f"Vocab used:            {len(tok):,} entries, built from training subset only (verified unchanged)")

section("1. TOKENS ENCOUNTERED")
print(f"Total tokens:          {len(ids):,}")
print(f"  real word/punct:     {num_real:,}")
print(f"  <EOS>:               {num_eos:,}")

section("2. <UNK> RATE")
print(f"<UNK> tokens:          {num_unk:,}")
print(f"  % of all tokens:     {100 * num_unk / len(ids):.3f}%")
print(f"  % of real tokens:    {100 * num_unk / num_real:.3f}%  (<EOS> can never be unknown)")
print(f"Distinct unknown words: {len(unk_words):,}")

section("3. TOP 20 WORDS THAT BECAME <UNK>")
print(f"{'rank':>4}  {'word':<16} {'count':>5}")
for rank, (w, c) in enumerate(unk_words.most_common(20), start=1):
    print(f"{rank:>4}  {w!r:<16} {c:>5}")

section("4. EXAMPLE SENTENCE WITH <UNK>")
# Naive sentence split: break after . ! ? (optionally followed by a quote).
sentences = [s for line in sample.splitlines() for s in re.split(r'(?<=[.!?])\s+|(?<=[.!?]")\s+', line)]
for s in sentences:
    s = s.strip()
    s_ids = tok.encode(s)
    if tok.unk_id in s_ids and len(s_ids) <= 25:
        break
s_tokens = split_into_tokens(s)
print(f"Before:  {s!r}\n")
print(f"{'token':<14} {'ID':>6}   decoded")
for t, i in zip(s_tokens, s_ids):
    marker = "   <-- not in training vocab" if i == tok.unk_id else ""
    print(f"{t!r:<14} {i:>6}   {tok.id_to_token[i]}{marker}")
print(f"\nIDs:     {s_ids}")
print(f"After:   {tok.decode(s_ids)!r}")
