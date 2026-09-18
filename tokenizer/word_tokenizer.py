"""
Stage 2 — a naive WORD-LEVEL tokenizer.

This is a deliberately simple tokenizer, built as a teaching step. It is NOT
the tokenizer the final model will use. The point is to see its weaknesses
first-hand (vocabulary explosion, out-of-vocabulary words) before we build
a BPE tokenizer later.

The whole idea in one sentence: split text into words and punctuation, and
give every distinct piece its own integer ID.

Run from the project root:
    venv/bin/python tokenizer/word_tokenizer.py
"""

import re
from collections import Counter
from pathlib import Path

# Resolve paths relative to this file, so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "tinystories_train_subset.txt"
# NOTE: data/tinystories_valid.txt is intentionally never touched here.

# The marker TinyStories puts between stories.
STORY_SEPARATOR = "<|endoftext|>"

# Our two special tokens.
UNK = "<UNK>"  # stands in for any token that is not in the vocabulary
EOS = "<EOS>"  # "end of sequence": replaces every <|endoftext|> marker

# The splitting rule. At each position in the text, the regex tries these
# three alternatives, left to right, and takes the first that matches:
#
#   <\|endoftext\|>   the story separator, matched whole, so it is not
#                     shredded into "<", "|", "endoftext", "|", ">"
#   \w+               a run of letters/digits           -> one word token
#   [^\w\s]           one char that is neither a letter/digit nor
#                     whitespace                        -> one punctuation token
#
# Whitespace matches none of them, so it is silently skipped. That means
# spaces and newlines are thrown away — remember this when we look at decode.
TOKEN_PATTERN = re.compile(r"<\|endoftext\|>|\w+|[^\w\s]")


def split_into_tokens(text):
    """Turn raw text into a list of token strings. Case is left untouched."""
    tokens = TOKEN_PATTERN.findall(text)
    # Swap the dataset's separator for our own <EOS> special token.
    return [EOS if tok == STORY_SEPARATOR else tok for tok in tokens]


class WordTokenizer:
    """Maps word/punctuation tokens to integer IDs and back."""

    def __init__(self, token_counts):
        """
        Build the vocabulary from a Counter of {token: count}.

        ID layout:
            0          -> <UNK>
            1          -> <EOS>
            2, 3, ...  -> real tokens, most frequent first
        """
        # Most frequent first. Ties are broken alphabetically so the IDs come
        # out identical every time the script runs.
        real_tokens = sorted(token_counts, key=lambda tok: (-token_counts[tok], tok))
        vocab = [UNK, EOS] + real_tokens

        # The two lookup tables: token -> ID (for encoding) and ID -> token
        # (for decoding).
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(vocab)}
        self.unk_id = self.token_to_id[UNK]

    def __len__(self):
        return len(self.token_to_id)

    def encode(self, text):
        """Text -> list of IDs. Unknown tokens become the <UNK> ID."""
        return [self.token_to_id.get(tok, self.unk_id) for tok in split_into_tokens(text)]

    def decode(self, ids):
        """
        List of IDs -> text.

        We have no record of where the original spaces were (the splitter
        threw them away), so the best we can do is put one space between
        every pair of tokens.
        """
        return " ".join(self.id_to_token[i] for i in ids)


# ---------------------------------------------------------------------------
# Everything below just builds the tokenizer and prints a report.
# ---------------------------------------------------------------------------

def section(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main():
    # --- 1. Read the training text -----------------------------------------
    text = TRAIN_FILE.read_text(encoding="utf-8")

    # --- 2. Split into tokens ----------------------------------------------
    tokens = split_into_tokens(text)
    num_stories = tokens.count(EOS)

    # --- 3. Count every unique real token (special tokens excluded) --------
    counts = Counter(tok for tok in tokens if tok != EOS)
    total_real_tokens = sum(counts.values())

    # --- 4-6. Build the vocabulary and both mappings -----------------------
    tokenizer = WordTokenizer(counts)

    # --- 7. Report ----------------------------------------------------------
    section("CORPUS")
    print(f"File:                    {TRAIN_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Characters:              {len(text):,}")
    print(f"Stories (<EOS> markers): {num_stories:,}")
    print(f"Real tokens (running):   {total_real_tokens:,}")

    section("VOCABULARY")
    print(f"Unique tokens found (before special tokens): {len(counts):,}")
    print(f"Final vocabulary size (incl. <UNK>, <EOS>):  {len(tokenizer):,}")
    print(f"  <UNK> -> ID {tokenizer.token_to_id[UNK]}")
    print(f"  <EOS> -> ID {tokenizer.token_to_id[EOS]}")

    section("30 MOST FREQUENT TOKENS")
    print(f"{'rank':>4}  {'ID':>4}  {'token':<10} {'count':>10}  {'% of all':>8}")
    for rank, (tok, count) in enumerate(counts.most_common(30), start=1):
        share = 100 * count / total_real_tokens
        print(f"{rank:>4}  {tokenizer.token_to_id[tok]:>4}  {tok!r:<10} {count:>10,}  {share:>7.2f}%")

    section("CASE SENSITIVITY (the 'The' vs 'the' lesson)")
    for tok in ["The", "the", "Lily", "lily"]:
        if tok in counts:
            print(f"  {tok!r:<7} ID {tokenizer.token_to_id[tok]:>5}   count {counts[tok]:>9,}")
        else:
            print(f"  {tok!r:<7} not in vocabulary")
    merged = len(counts) - len({tok.lower() for tok in counts})
    print(f"Entries that exist only because of capitalisation: {merged:,}")

    section("RARE TOKENS (the vocabulary-explosion lesson)")
    singletons = sorted(tok for tok, c in counts.items() if c == 1)
    print(f"Tokens seen exactly once: {len(singletons):,} "
          f"({100 * len(singletons) / len(counts):.1f}% of the vocabulary)")
    print(f"A few of them: {singletons[:: max(1, len(singletons) // 12)][:12]}")

    section("<UNK> CHECK: encoding the training data with its own vocab")
    train_ids = tokenizer.encode(text)
    num_unk = train_ids.count(tokenizer.unk_id)
    print(f"Tokens encoded: {len(train_ids):,}")
    print(f"<UNK> count:    {num_unk:,}")
    if num_unk == 0:
        print("OK — zero unknowns, as expected (vocab was built from this exact text).")
    else:
        print("WARNING — expected zero unknowns here. Something is inconsistent!")

    section("ENCODE / DECODE TEST")
    sentence = "One day, Lily found a needle."
    ids = tokenizer.encode(sentence)
    decoded = tokenizer.decode(ids)

    print(f"{'pos':>3}  {'token':<9} {'ID':>6}   {'decoded':<9}")
    for pos, (tok, i) in enumerate(zip(split_into_tokens(sentence), ids)):
        print(f"{pos:>3}  {tok!r:<9} {i:>6}   {tokenizer.id_to_token[i]!r:<9}")
    print()
    print(f"Original: {sentence!r}")
    print(f"IDs:      {ids}")
    print(f"Decoded:  {decoded!r}")
    print()
    print(f"Same tokens after round trip? {split_into_tokens(decoded) == split_into_tokens(sentence)}")
    print(f"Exact same string?            {decoded == sentence}")


if __name__ == "__main__":
    main()
