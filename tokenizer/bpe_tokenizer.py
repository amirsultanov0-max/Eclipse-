"""
Stage 2b — a character-level BPE (Byte Pair Encoding) tokenizer, from scratch.

The whole algorithm:
  1. Start with a vocabulary of single characters.
  2. Count every pair of adjacent symbols in the training text.
  3. Merge the most frequent pair into one new symbol; add it to the vocab.
  4. Repeat until the vocabulary reaches the target size.

Frequent words end up as one token (" the"), rarer words as a few pieces
(" thirst" + "ier"), and any word at all can fall back to single characters,
so there is no such thing as an unknown WORD any more.

Train it (writes tokenizer/bpe_4000.json):
    venv/bin/python tokenizer/bpe_tokenizer.py
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "data" / "tinystories_train_subset.txt"
SAVE_FILE = PROJECT_ROOT / "tokenizer" / "bpe_4000.json"
TARGET_VOCAB_SIZE = 4000

STORY_SEPARATOR = "<|endoftext|>"
UNK = "<UNK>"  # only for CHARACTERS never seen in training (e.g. "é", emoji)
EOS = "<EOS>"  # one per "<|endoftext|>" marker
SPECIAL_TOKENS = [UNK, EOS]  # IDs 0 and 1, same as the word-level tokenizer

# Pre-tokenization: before BPE runs, text is cut into chunks, and merges are
# never allowed to cross a chunk boundary (so no token can span two words).
# Each alternative, tried left to right:
#
#   ['’](?:s|t|re|ve|m|ll|d)  English contractions: "'s", "'t", "'ll", ...
#                           with either a straight (') or curly (’) apostrophe,
#                           since the corpus uses both
#    ?[A-Za-z]+             a word, WITH the space in front of it: " cat"
#    ?[0-9]+                a number, with its leading space
#    ?[^\sA-Za-z0-9]+       a run of punctuation, with its leading space: '."'
#   \s+(?!\S)|\s+           whitespace not already attached to a word ("\n")
#
# Every character matches one of these, so nothing is thrown away. That is
# what makes decoding exact — unlike the word-level tokenizer, which dropped
# all whitespace. It also means " cat" (a word start) and "cat" (inside
# "concatenate") are different symbols, which is useful information.
# This is a simplified version of GPT-2's pre-tokenization rule.
#
# Known limitation: "letters" means A-Z only, so an accented word like "café"
# is split at the accent (" caf" + "é"). Accented letters occur only 13 times
# in the training subset, so this is deliberately left as is.
CHUNK_PATTERN = re.compile(r"['’](?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def split_into_chunks(text):
    """Cut text (containing no story separators) into pre-tokenized chunks."""
    return CHUNK_PATTERN.findall(text)


def merge_pair(symbols, pair, new_symbol):
    """
    Replace every adjacent occurrence of `pair` in `symbols` with `new_symbol`,
    scanning left to right.
        merge_pair(["t", "h", "e"], ("t", "h"), "th")  ->  ["th", "e"]
    """
    out = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_symbol)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class BPETokenizer:
    """
    A trained BPE tokenizer is fully described by two things:
      - the alphabet (the starting single characters), and
      - the ordered list of merges it learned.
    Everything else (the vocab, the ID mappings) is derived from those.
    """

    def __init__(self, alphabet, merges, merge_counts=None):
        self.alphabet = alphabet
        self.merges = merges  # list of (left, right) pairs, in learned order
        self.merge_counts = merge_counts  # pair frequency when it was merged (for inspection only)

        # Rank = position in the merge list. When encoding, the pair with the
        # LOWEST rank (learned earliest) is always merged first.
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

        # Vocab layout: specials, then single characters, then merged tokens in
        # the order they were learned. dict.fromkeys drops the rare case where
        # two different merges build the same string (e.g. "th"+"e" and "t"+"he").
        vocab = list(dict.fromkeys(SPECIAL_TOKENS + alphabet + [a + b for a, b in merges]))
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(vocab)}
        self.unk_id = self.token_to_id[UNK]
        self.eos_id = self.token_to_id[EOS]

        # The same chunk (" the") appears millions of times; encode it once.
        self._chunk_cache = {}

    def __len__(self):
        return len(self.token_to_id)

    # --- Encoding -----------------------------------------------------------

    def encode_chunk(self, chunk):
        """
        Encode one pre-tokenized chunk by replaying the learned merges.

        Start from single characters. Repeatedly look at every adjacent pair,
        pick the one that was learned EARLIEST in training, and merge it.
        Stop when no adjacent pair is a learned merge.
        """
        if chunk in self._chunk_cache:
            return self._chunk_cache[chunk]

        symbols = list(chunk)
        while len(symbols) > 1:
            pairs = set(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda p: self.merge_rank.get(p, float("inf")))
            if best not in self.merge_rank:
                break  # none of the remaining pairs was ever learned
            symbols = merge_pair(symbols, best, best[0] + best[1])

        # A symbol is only missing from the vocab if it's a character that
        # never appeared in the training text.
        ids = [self.token_to_id.get(s, self.unk_id) for s in symbols]
        self._chunk_cache[chunk] = ids
        return ids

    def encode(self, text):
        """Text -> list of IDs. Each "<|endoftext|>" marker becomes <EOS>."""
        ids = []
        for i, piece in enumerate(text.split(STORY_SEPARATOR)):
            if i > 0:
                ids.append(self.eos_id)
            for chunk in split_into_chunks(piece):
                ids.extend(self.encode_chunk(chunk))
        return ids

    def decode(self, ids):
        """
        IDs -> text. Tokens carry their own spaces and newlines, so we simply
        glue them together. <EOS> is written back as "<|endoftext|>", so
        decode(encode(text)) == text exactly.
        """
        return "".join(STORY_SEPARATOR if i == self.eos_id else self.id_to_token[i] for i in ids)

    # --- Saving / loading ---------------------------------------------------

    def save(self, path):
        data = {
            "special_tokens": SPECIAL_TOKENS,
            "alphabet": self.alphabet,
            "merges": [list(pair) for pair in self.merges],
            "merge_counts": self.merge_counts,
        }
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["alphabet"], [tuple(p) for p in data["merges"]], data.get("merge_counts"))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(text, target_vocab_size, log_every=500):
    """Learn merges from `text` until the vocab has `target_vocab_size` entries."""

    # 1. Chunk the text and count each distinct chunk. BPE then works on the
    #    ~tens of thousands of DISTINCT chunks (weighted by their counts)
    #    instead of millions of running chunks — same result, far faster.
    chunk_counts = Counter()
    for story in text.split(STORY_SEPARATOR):
        chunk_counts.update(split_into_chunks(story))

    # 2. The starting vocabulary: every character that appears.
    alphabet = sorted({ch for chunk in chunk_counts for ch in chunk})

    # 3. Every distinct chunk starts out as a list of single characters.
    words = [list(chunk) for chunk in chunk_counts]
    freqs = list(chunk_counts.values())

    # 4. Count every adjacent pair, weighted by how often its chunk occurs.
    #    Also remember WHICH words contain each pair, so that after a merge we
    #    only revisit those words instead of rescanning everything.
    pair_counts = Counter()
    words_with_pair = defaultdict(set)
    for idx, symbols in enumerate(words):
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += freqs[idx]
            words_with_pair[pair].add(idx)

    print(f"Distinct chunks: {len(words):,}   alphabet: {len(alphabet)} characters")

    merges, merge_counts = [], []
    vocab = set(SPECIAL_TOKENS) | set(alphabet)
    start = time.time()

    while len(vocab) < target_vocab_size and pair_counts:
        # 5. The most frequent pair wins. It becomes a new vocab entry.
        best = max(pair_counts, key=pair_counts.__getitem__)
        new_symbol = best[0] + best[1]
        merges.append(best)
        merge_counts.append(pair_counts[best])
        vocab.add(new_symbol)

        # 6. Apply the merge inside every word that contains the pair, and
        #    update the pair counts for just those words: remove the old
        #    word's pairs, add the new word's pairs.
        for idx in words_with_pair.pop(best):
            old = words[idx]
            new = merge_pair(old, best, new_symbol)
            if len(new) == len(old):
                continue  # stale entry: this word no longer contains the pair
            freq = freqs[idx]
            for pair in zip(old, old[1:]):
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            for pair in zip(new, new[1:]):
                pair_counts[pair] += freq
                words_with_pair[pair].add(idx)
            words[idx] = new

        if len(merges) % log_every == 0:
            print(f"  merge {len(merges):>5}: {best[0]!r} + {best[1]!r} -> {new_symbol!r}  "
                  f"(count {merge_counts[-1]:,})   vocab {len(vocab):,}   {time.time() - start:.1f}s")

    print(f"Done: {len(merges):,} merges, vocab {len(vocab):,}, {time.time() - start:.1f}s")
    return BPETokenizer(alphabet, merges, merge_counts)


def main():
    text = TRAIN_FILE.read_text(encoding="utf-8")
    print(f"Training on {TRAIN_FILE.relative_to(PROJECT_ROOT)} ({len(text):,} chars), "
          f"target vocab {TARGET_VOCAB_SIZE:,}")
    tokenizer = train(text, TARGET_VOCAB_SIZE)
    tokenizer.save(SAVE_FILE)
    print(f"Saved to {SAVE_FILE.relative_to(PROJECT_ROOT)} (final vocab: {len(tokenizer):,})")


if __name__ == "__main__":
    main()
