"""
Stage 6.1, step 2: build the training token stream from the 200 MB corpus.

Implements docs/stage6_1_design.md section 7.

  1. Reconstruct the baseline's 95/5 split deterministically (seed 1337 over the
     22,174 baseline stories) and take the same 1,109 monitoring stories.
  2. Verify the re-encoded monitoring stream is byte-identical to the existing
     data/stage3_val_tokens.npy. Abort if not — the held-out set must not move.
  3. Training set = every corpus story except the monitoring split, except exact
     duplicates of a monitoring story, and except exact duplicates of an official
     validation story (decision 5).
  4. Write data/stage6_1_train_tokens.npy. The monitoring stream is reused as is,
     not regenerated.

    venv/bin/python prepare_data_stage6.py
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from tokenizer.bpe_tokenizer import SAVE_FILE, STORY_SEPARATOR, BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_FILE = PROJECT_ROOT / "data" / "tinystories_train_200mb.txt"
BASELINE_SUBSET = PROJECT_ROOT / "data" / "tinystories_train_subset.txt"
OFFICIAL_VALID = PROJECT_ROOT / "data" / "tinystories_valid.txt"
MONITORING_TOKENS = PROJECT_ROOT / "data" / "stage3_val_tokens.npy"
OUT_TRAIN = PROJECT_ROOT / "data" / "stage6_1_train_tokens.npy"
MANIFEST = PROJECT_ROOT / "metadata" / "stage6_1_corpus.json"

VAL_FRACTION = 0.05
SPLIT_SEED = 1337
FLUSH_EVERY = 20_000        # stories per numpy chunk, to keep memory flat


def split_stories(text):
    return [s for s in (part.strip("\n") for part in text.split(STORY_SEPARATOR)) if s]


def story_hash(story):
    return hashlib.sha256(story.encode("utf-8")).hexdigest()


def encode_stream(tokenizer, stories):
    """Encode stories into one uint16 stream, one <EOS> after each."""
    chunks, buffer = [], []
    for i, story in enumerate(stories, start=1):
        buffer.extend(tokenizer.encode(story))
        buffer.append(tokenizer.eos_id)
        if i % FLUSH_EVERY == 0:
            chunks.append(np.array(buffer, dtype=np.uint16))
            buffer = []
    if buffer:
        chunks.append(np.array(buffer, dtype=np.uint16))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)


def main():
    tokenizer = BPETokenizer.load(SAVE_FILE)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # --- 1. reconstruct the baseline split ---------------------------------
    baseline_stories = split_stories(BASELINE_SUBSET.read_text(encoding="utf-8"))
    shuffled = list(baseline_stories)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    num_val = round(len(shuffled) * VAL_FRACTION)
    monitoring = shuffled[:num_val]
    monitoring_hashes = {story_hash(s) for s in monitoring}
    print(f"baseline stories {len(baseline_stories):,} -> monitoring {len(monitoring):,}")

    # --- 2. verify the monitoring split has not moved ----------------------
    rebuilt = encode_stream(tokenizer, monitoring)
    existing = np.load(MONITORING_TOKENS)
    identical = bool(np.array_equal(rebuilt, existing))
    print(f"monitoring stream identical to {MONITORING_TOKENS.name}: {identical} "
          f"({len(existing):,} tokens)")
    assert identical, "monitoring split changed — aborting"

    # --- 3. build the training story list ----------------------------------
    corpus = split_stories(CORPUS_FILE.read_text(encoding="utf-8"))
    valid_hashes = {story_hash(s) for s in split_stories(
        OFFICIAL_VALID.read_text(encoding="utf-8"))}

    training, skipped_monitoring, removed_valid = [], 0, 0
    for story in corpus:
        h = story_hash(story)
        if h in monitoring_hashes:
            skipped_monitoring += 1
            continue
        if h in valid_hashes:
            removed_valid += 1
            continue
        training.append(story)
    print(f"corpus {len(corpus):,} stories -> training {len(training):,} "
          f"(monitoring/dupes excluded {skipped_monitoring:,}, "
          f"validation duplicates removed {removed_valid:,})")

    expected = manifest["leakage"]["accounting"]["training_stories"]
    assert len(training) == expected, f"expected {expected:,} training stories"

    # --- 4. encode and save ------------------------------------------------
    random.Random(SPLIT_SEED).shuffle(training)
    start = time.time()
    stream = encode_stream(tokenizer, training)
    np.save(OUT_TRAIN, stream)
    print(f"training stream {len(stream):,} tokens in {time.time() - start:.0f}s "
          f"-> {OUT_TRAIN.relative_to(PROJECT_ROOT)} "
          f"({OUT_TRAIN.stat().st_size / 1e6:.1f} MB)")

    expected_tokens = manifest["tokenizer_audit"]["training_bpe_tokens"]
    assert len(stream) == expected_tokens, f"expected {expected_tokens:,} tokens"
    print(f"matches the manifest: {expected_tokens:,} tokens, {expected:,} stories")

    eos = int((stream == tokenizer.eos_id).sum())
    unk = int((stream == tokenizer.unk_id).sum())
    print(f"<EOS> {eos:,} (one per story: {eos == len(training)}) | <UNK> {unk:,}")
    print(f"\nbaseline train stream: {len(np.load(PROJECT_ROOT / 'data' / 'stage3_train_tokens.npy')):,} tokens")
    print(f"stage 6.1 train stream: {len(stream):,} tokens "
          f"({len(stream) / 4_691_267:.1f}x)")
    print(f"epochs at the 81,920,000-token budget: {81_920_000 / len(stream):.2f} "
          f"(baseline 17.46)")


if __name__ == "__main__":
    main()
