"""
Read-only report on the trained BPE tokenizer (tokenizer/bpe_4000.json).

Nothing is trained or modified here. It reads:
  - the saved merges,
  - the training subset (vocab composition and per-story statistics),
  - the first 200 validation stories (OOV check only — the same slice that
    tokenizer/inspect_unk_rate.py measured for the word-level tokenizer).

Run from the project root:
    venv/bin/python tokenizer/inspect_bpe.py
"""

import statistics
from collections import Counter

import word_tokenizer as wt
from bpe_tokenizer import (PROJECT_ROOT, SAVE_FILE, SPECIAL_TOKENS, STORY_SEPARATOR,
                           TRAIN_FILE, BPETokenizer, split_into_chunks)

VALID_FILE = PROJECT_ROOT / "data" / "tinystories_valid.txt"
NUM_VALID_STORIES = 200
CONTEXT_WINDOWS = [128, 256, 512, 1024]


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def pieces(tok, text):
    """Show how BPE splits `text`, e.g. ' thirst' | 'ier'."""
    return " | ".join(repr(tok.id_to_token[i]) for i in tok.encode(text))


def percentile(values, p):
    return statistics.quantiles(values, n=100)[p - 1]


tok = BPETokenizer.load(SAVE_FILE)
train_text = TRAIN_FILE.read_text(encoding="utf-8")

# Stories, with the newline on either side of each separator trimmed off.
stories = [s.strip("\n") for s in train_text.split(STORY_SEPARATOR)]
stories = [s for s in stories if s]

# Every complete chunk (" the", "Once", ...) that occurs in the training text.
train_chunks = {c for s in stories for c in split_into_chunks(s)}

# =========================================================================
section("TOKENIZER")
real_tokens = [t for t in tok.token_to_id if t not in SPECIAL_TOKENS]
print(f"Vocab size:        {len(tok):,}")
print(f"  special tokens:  {len(SPECIAL_TOKENS)}  (<UNK>=0, <EOS>=1)")
print(f"  alphabet:        {len(tok.alphabet)} characters")
print(f"  merged tokens:   {len(real_tokens) - len(tok.alphabet):,}  (from {len(tok.merges):,} merges)")
missing = sorted(set(map(chr, range(32, 127))) - set(tok.alphabet))
print(f"Alphabet includes newline: {'\n' in tok.alphabet}.  Printable ASCII never seen in training: {missing}")

# =========================================================================
section("4. VOCABULARY COMPOSITION")


def category(t):
    if len(t) == 1:
        return "single character"
    if t.lstrip(" ").isalpha():
        # A word normally starts with a space (" cat"), or with a capital at
        # the start of a line ("Once"). Lowercase tokens with no space ("ing",
        # "ll") are pieces of longer words, even though a few of them
        # occasionally appear on their own in the text.
        starts_word = t.startswith(" ") or t[0].isupper()
        return "whole word" if starts_word and t in train_chunks else "word piece"
    return "other (punctuation, contractions, whitespace, digits)"


by_category = {}
for t in real_tokens:
    by_category.setdefault(category(t), []).append(t)

for name in ["single character", "word piece", "whole word",
             "other (punctuation, contractions, whitespace, digits)"]:
    group = by_category.get(name, [])
    sample = group[:: max(1, len(group) // 10)][:10]
    print(f"{name:<55} {len(group):>5}  ({100 * len(group) / len(real_tokens):.1f}%)")
    print(f"    e.g. {', '.join(repr(t) for t in sample)}")

print("\nFirst 15 merges learned (most frequent pairs in the data):")
for rank in range(15):
    a, b = tok.merges[rank]
    print(f"  #{rank + 1:<5} {a!r:>8} + {b!r:<8} -> {a + b!r:<10} count {tok.merge_counts[rank]:>9,}")

print("\nEvery 300th merge after that:")
for rank in range(299, len(tok.merges), 300):
    a, b = tok.merges[rank]
    print(f"  #{rank + 1:<5} {a!r:>10} + {b!r:<8} -> {a + b!r:<14} count {tok.merge_counts[rank]:>9,}")

longest = sorted(real_tokens, key=len, reverse=True)[:10]
print(f"\nLongest tokens: {', '.join(repr(t) for t in longest)}")

print("\nDuplication still present in the BPE vocab:")
case_dupes = len(real_tokens) - len({t.lower() for t in real_tokens})
space_twins = sum(1 for t in real_tokens if len(t) > 1 and t.startswith(" ") and t[1:] in tok.token_to_id)
print(f"  entries that exist only because of capitalisation: {case_dupes:,} of {len(real_tokens):,} "
      f"({100 * case_dupes / len(real_tokens):.1f}%)   [word-level: 1,813 of 12,994 (14.0%)]")
print(f"  entries that also exist without their leading space: {space_twins:,}")
for family in [["the", " the", "The", " The"], ["lily", " lily", "Lily", " Lily"]]:
    present = {t: tok.token_to_id[t] for t in family if t in tok.token_to_id}
    print(f"  family {family[0]!r}: {present}")

# =========================================================================
section("5. COMPRESSION: TOKENS PER TRAINING STORY")
whitespace_ids = {i for t, i in tok.token_to_id.items() if t.isspace()}
bpe_lengths, word_lengths, newline_tokens = [], [], 0
for story in stories:
    ids = tok.encode(story)
    bpe_lengths.append(len(ids))
    newline_tokens += sum(1 for i in ids if i in whitespace_ids)
    word_lengths.append(len(wt.split_into_tokens(story)))

story_chars = sum(len(s) for s in stories)
print(f"Stories: {len(stories):,}   (word-level total check: {sum(word_lengths):,} — expected 4,683,445)\n")
print(f"{'':<22} {'word-level':>12} {'BPE (4,000)':>12}")
rows = [
    ("total tokens", sum(word_lengths), sum(bpe_lengths)),
    ("average per story", statistics.mean(word_lengths), statistics.mean(bpe_lengths)),
    ("median", statistics.median(word_lengths), statistics.median(bpe_lengths)),
    ("shortest", min(word_lengths), min(bpe_lengths)),
    ("longest", max(word_lengths), max(bpe_lengths)),
    ("90th percentile", percentile(word_lengths, 90), percentile(bpe_lengths, 90)),
    ("99th percentile", percentile(word_lengths, 99), percentile(bpe_lengths, 99)),
    ("characters per token", story_chars / sum(word_lengths), story_chars / sum(bpe_lengths)),
]
for label, w, b in rows:
    fmt = (lambda v: f"{v:>12,.2f}") if isinstance(w, float) or isinstance(b, float) else (lambda v: f"{v:>12,}")
    print(f"{label:<22} {fmt(w)} {fmt(b)}")
print(f"\nBPE / word-level token ratio: {sum(bpe_lengths) / sum(word_lengths):.3f}")
print(f"Whitespace tokens inside BPE counts (paragraph breaks; word-level drops these): {newline_tokens:,} "
      f"({newline_tokens / len(stories):.2f} per story)")

# =========================================================================
section(f"6. OOV CHECK: FIRST {NUM_VALID_STORIES} VALIDATION STORIES (read-only)")
lines, seen = [], 0
with VALID_FILE.open(encoding="utf-8") as f:
    for line in f:
        lines.append(line)
        if line.strip() == STORY_SEPARATOR:
            seen += 1
            if seen == NUM_VALID_STORIES:
                break
sample = "".join(lines)

ids = tok.encode(sample)
num_unk = ids.count(tok.unk_id)
unseen_chars = set(sample.replace(STORY_SEPARATOR, "")) - set(tok.alphabet)
print(f"Stories read: {ids.count(tok.eos_id)} (lines 1-{len(lines)})")
print(f"Total BPE tokens:  {len(ids):,}  (word-level on same slice: 37,165)")
print(f"<UNK> tokens:      {num_unk}  ({100 * num_unk / len(ids):.3f}%)   [word-level: 49 (0.132%)]")
print(f"Characters in the slice never seen in training: {sorted(unseen_chars) or 'none'}")
print(f"Exact round trip decode(encode(text)) == text: {tok.decode(ids) == sample}")

# Rebuild the word-level vocab (from the training subset, read-only) to list
# exactly which words it could not encode.
word_counts = Counter(t for t in wt.split_into_tokens(train_text) if t != wt.EOS)
word_tok = wt.WordTokenizer(word_counts)
wl_unknown = Counter(t for t in wt.split_into_tokens(sample) if t not in word_tok.token_to_id)
sample_chunks = Counter(c for piece in sample.split(STORY_SEPARATOR) for c in split_into_chunks(piece))

print(f"\nThe {len(wl_unknown)} words that were <UNK> for the word-level tokenizer, now under BPE:")
print(f"  {'word':<13} {'as it appears':<16} {'count':>5}   BPE pieces")
for word, _ in wl_unknown.most_common():
    forms = [c for c, _ in sample_chunks.most_common() if c.lstrip(" ") == word]
    for form in forms:
        n = len(tok.encode(form))
        print(f"  {word:<13} {form!r:<16} {sample_chunks[form]:>5}   {pieces(tok, form)}   ({n} tokens)")

print("\nThe two tracked failure cases, with IDs:")
for form in [" thirstier", "Roxy", " Roxy"]:
    enc = tok.encode(form)
    print(f"  {form!r:<13} -> {[tok.id_to_token[i] for i in enc]}  IDs {enc}")

sentence = "Once upon a time, in a big forest, there lived a rhinoceros named Roxy."
enc = tok.encode(sentence)
print(f"\nSame example sentence as the word-level check:")
print(f"  Before:  {sentence!r}")
print(f"  Tokens:  {[tok.id_to_token[i] for i in enc]}")
print(f"  IDs:     {enc}")
print(f"  After:   {tok.decode(enc)!r}")
print(f"  Exact match: {tok.decode(enc) == sentence}   ({len(enc)} BPE tokens vs 17 word-level)")

# =========================================================================
section("7. CONTEXT WINDOW FIT (training stories; a story needs its tokens + 1 <EOS>)")
print(f"{'window':>8} {'stories that fit':>18} {'share':>8}")
for n in CONTEXT_WINDOWS:
    fits = sum(1 for length in bpe_lengths if length + 1 <= n)
    print(f"{n:>8} {fits:>18,} {100 * fits / len(stories):>7.2f}%")
sorted_lengths = sorted(bpe_lengths)
for p in [50, 90, 95, 99]:
    # smallest window that holds at least p% of stories (+1 for <EOS>)
    need = sorted_lengths[-(-p * len(sorted_lengths) // 100) - 1] + 1
    print(f"  window needed to fit {p}% of stories whole: {need:,} tokens")
