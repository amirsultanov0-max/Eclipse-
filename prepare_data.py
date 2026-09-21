"""
Stage 3, step 1: turn the TinyStories training subset into two token streams.

Each story is encoded with the Stage 2b BPE tokenizer and followed by <EOS>,
then the stories are concatenated into one long stream of token IDs:

    story A ... <EOS> story B ... <EOS> story C ...

Training examples are just neighbouring pairs in that stream: stream[i]
predicts stream[i+1]. The pair that straddles a story boundary (last token of
a story -> <EOS>, and <EOS> -> first token of the next story) is kept on
purpose: it teaches the model how stories end and begin.

The 22,174 stories are shuffled with a fixed seed and split 95% / 5%. The
split is by STORY, not by token, so no story appears in both streams.
data/tinystories_valid.txt is not touched — it is reserved for later stages.

Run from the project root:
    venv/bin/python prepare_data.py
"""

import random

import numpy as np

from tokenizer.bpe_tokenizer import BPETokenizer, SAVE_FILE, STORY_SEPARATOR, TRAIN_FILE

PROJECT_ROOT = TRAIN_FILE.parent.parent
OUT_TRAIN = PROJECT_ROOT / "data" / "stage3_train_tokens.npy"
OUT_VAL = PROJECT_ROOT / "data" / "stage3_val_tokens.npy"

VAL_FRACTION = 0.05
SEED = 1337


def build_stream(tokenizer, stories):
    """Encode every story, append <EOS> after each, and concatenate."""
    ids = []
    for story in stories:
        ids.extend(tokenizer.encode(story))
        ids.append(tokenizer.eos_id)
    # uint16 is enough for a 4,000-token vocabulary and keeps the files small.
    return np.array(ids, dtype=np.uint16)


def main():
    tokenizer = BPETokenizer.load(SAVE_FILE)
    assert len(tokenizer) < 2**16, "uint16 storage needs a vocab smaller than 65,536"

    text = TRAIN_FILE.read_text(encoding="utf-8")
    stories = [s.strip("\n") for s in text.split(STORY_SEPARATOR)]
    stories = [s for s in stories if s]

    # Shuffle before splitting, so the validation stories are not just the
    # tail of the file. The fixed seed makes the split reproducible.
    random.Random(SEED).shuffle(stories)
    num_val = round(len(stories) * VAL_FRACTION)
    val_stories, train_stories = stories[:num_val], stories[num_val:]

    train_stream = build_stream(tokenizer, train_stories)
    val_stream = build_stream(tokenizer, val_stories)

    np.save(OUT_TRAIN, train_stream)
    np.save(OUT_VAL, val_stream)

    print(f"Tokenizer: {SAVE_FILE.name}, vocab {len(tokenizer):,}, <EOS> id {tokenizer.eos_id}")
    print(f"Stories: {len(stories):,} total -> {len(train_stories):,} train / {len(val_stories):,} val\n")
    for name, stream, story_count, path in [
        ("train", train_stream, len(train_stories), OUT_TRAIN),
        ("val", val_stream, len(val_stories), OUT_VAL),
    ]:
        eos_count = int((stream == tokenizer.eos_id).sum())
        print(f"{name:>5}: {len(stream):,} tokens   {len(stream) - 1:,} pairs   "
              f"{eos_count:,} <EOS> (one per story: {eos_count == story_count})   "
              f"mean {len(stream) / story_count:.1f} tokens/story")
        print(f"       -> {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")

    # Visual check: decode the start of the training stream back to text.
    print("\nFirst 60 tokens of the training stream, decoded:")
    print(repr(tokenizer.decode([int(i) for i in train_stream[:60]])))


if __name__ == "__main__":
    main()
