"""
Stage 6.1, step 1: build the ~200 MB training corpus and its manifest.

Implements docs/stage6_1_design.md sections 5-6. Read-only with respect to every
existing artefact: it appends to a new corpus file, and touches
data/tinystories_valid.txt only to hash its stories for the leakage check.

    venv/bin/python fetch_corpus_stage6.py
"""

import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tokenizer.bpe_tokenizer import SAVE_FILE, STORY_SEPARATOR, BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_SUBSET = PROJECT_ROOT / "data" / "tinystories_train_subset.txt"
OFFICIAL_VALID = PROJECT_ROOT / "data" / "tinystories_valid.txt"
MONITORING_TOKENS = PROJECT_ROOT / "data" / "stage3_val_tokens.npy"
CHUNK_FILE = PROJECT_ROOT / "data" / "_stage6_chunk.bin"
CORPUS_FILE = PROJECT_ROOT / "data" / "tinystories_train_200mb.txt"
MANIFEST = PROJECT_ROOT / "metadata" / "stage6_1_corpus.json"

REPO_SHA = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
       f"{REPO_SHA}/TinyStories-train.txt")
TARGET_TOTAL_BYTES = 200_000_000     # decimal MB, per the design
PROBE = 4096
SEPARATOR_BYTES = STORY_SEPARATOR.encode()

VAL_FRACTION = 0.05                  # identical to prepare_data.py
SPLIT_SEED = 1337
TOKEN_BUDGET = 10_000 * 16 * 512     # 81,920,000 tokens, both runs


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def story_hash(story):
    return hashlib.sha256(story.encode("utf-8")).hexdigest()


def fetch_range(start, end, out_path):
    """HTTP Range request against the pinned revision. Returns (code, bytes)."""
    result = subprocess.run(
        ["curl", "-sL", "--fail", "--max-time", "1800", "-r", f"{start}-{end}",
         "-o", str(out_path), "-w", "%{http_code} %{size_download}", URL],
        capture_output=True, text=True, check=True)
    code, size = result.stdout.split()
    return int(code), int(size)


def split_stories(text):
    """Same normalisation as prepare_data.py: strip only leading/trailing newlines."""
    return [s for s in (part.strip("\n") for part in text.split(STORY_SEPARATOR)) if s]


def main():
    log = lambda msg: print(msg, flush=True)
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "design": "docs/stage6_1_design.md"}

    # --- 5.2 pin the source ------------------------------------------------
    head = subprocess.run(["curl", "-sIL", "--max-time", "120", URL],
                          capture_output=True, text=True, check=True).stdout
    headers = {}
    for line in head.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            headers.setdefault(key.strip().lower(), value.strip())
    manifest["source"] = {
        "url": URL, "revision_sha": REPO_SHA,
        "x_repo_commit": headers.get("x-repo-commit"),
        "etag": headers.get("etag"), "x_linked_etag": headers.get("x-linked-etag"),
        "content_length": int(headers.get("x-linked-size") or headers.get("content-length")),
        "accept_ranges": headers.get("accept-ranges"),
    }
    assert manifest["source"]["x_repo_commit"] == REPO_SHA, "pinned SHA mismatch"
    log(f"source pinned: {REPO_SHA} | {manifest['source']['content_length']:,} bytes")

    # --- 5.3 offset probes -------------------------------------------------
    local = BASELINE_SUBSET.read_bytes()
    S = len(local)
    code_h, _ = fetch_range(0, PROBE - 1, CHUNK_FILE)
    head_probe = CHUNK_FILE.read_bytes()
    code_s, _ = fetch_range(S - PROBE, S + PROBE - 1, CHUNK_FILE)
    seam_probe = CHUNK_FILE.read_bytes()
    head_ok = head_probe == local[:PROBE]
    seam_ok = seam_probe[:PROBE] == local[-PROBE:]
    after = seam_probe[PROBE:]
    boundary_ok = after[:1] == b"\n" and after[1:14] != SEPARATOR_BYTES
    log(f"head probe {head_ok} (HTTP {code_h}) | seam probe {seam_ok} (HTTP {code_s}) "
        f"| boundary {boundary_ok}")
    assert head_ok and seam_ok and boundary_ok, "offset verification failed — see design 5.3"
    manifest["offset"] = {"baseline_bytes": S, "method": "prefix-verified",
                          "head_probe_ok": head_ok, "seam_probe_ok": seam_ok,
                          "boundary_ok": boundary_ok, "probe_bytes": PROBE}

    # --- 5.4 fetch and trim ------------------------------------------------
    start = time.time()
    log(f"fetching bytes {S:,}-{TARGET_TOTAL_BYTES - 1:,} "
        f"({(TARGET_TOTAL_BYTES - S) / 1e6:.0f} MB)...")
    code, size = fetch_range(S, TARGET_TOTAL_BYTES - 1, CHUNK_FILE)
    chunk = CHUNK_FILE.read_bytes()
    assert code == 206 and size == len(chunk) == TARGET_TOTAL_BYTES - S, \
        f"unexpected response: HTTP {code}, {size} bytes"
    log(f"  fetched {len(chunk):,} bytes in {time.time() - start:.0f}s")

    cut = chunk.rfind(SEPARATOR_BYTES)
    assert cut != -1, "no separator found in the fetched chunk"
    trimmed = chunk[:cut + len(SEPARATOR_BYTES)]
    manifest["fetch"] = {
        "requested_range": [S, TARGET_TOTAL_BYTES - 1], "http_code": code,
        "received_bytes": len(chunk), "trim_end_offset": S + len(trimmed),
        "bytes_discarded_in_trim": len(chunk) - len(trimmed), "new_bytes": len(trimmed),
    }

    combined = local + trimmed
    CORPUS_FILE.write_bytes(combined)
    CHUNK_FILE.unlink()
    assert CORPUS_FILE.stat().st_size == S + len(trimmed)
    assert combined[:S] == local, "prefix corrupted during assembly"
    manifest["corpus_file"] = {
        "path": str(CORPUS_FILE.relative_to(PROJECT_ROOT)),
        "total_bytes": len(combined), "sha256": sha256(combined),
        "sha256_baseline_subset": sha256(local),
    }
    log(f"corpus assembled: {len(combined):,} bytes "
        f"({manifest['fetch']['bytes_discarded_in_trim']:,} discarded in trim)")

    # --- 5.5 seam and duplication -----------------------------------------
    text = combined.decode("utf-8")
    stories = split_stories(text)
    baseline_stories = split_stories(local.decode("utf-8"))
    n_base = len(baseline_stories)
    prefix_match = all(story_hash(a) == story_hash(b)
                       for a, b in zip(baseline_stories, stories[:n_base]))
    seam_distinct = stories[n_base] != stories[n_base - 1]
    seam_context = text[S - 60:S + 60]
    hashes = [story_hash(s) for s in stories]
    duplicate_count = len(hashes) - len(set(hashes))
    log(f"stories: {len(stories):,} total ({n_base:,} baseline + {len(stories) - n_base:,} new)")
    log(f"  prefix stories identical: {prefix_match} | seam stories distinct: {seam_distinct}"
        f" | exact duplicate stories in corpus: {duplicate_count:,}")
    assert prefix_match and seam_distinct, "seam verification failed"
    manifest["seam"] = {
        "baseline_stories": n_base, "new_stories": len(stories) - n_base,
        "total_stories": len(stories), "prefix_stories_identical": prefix_match,
        "seam_stories_distinct": seam_distinct,
        "seam_context": seam_context, "duplicate_story_texts_in_corpus": duplicate_count,
    }

    # --- 7. reconstruct and verify the monitoring split --------------------
    shuffled = list(baseline_stories)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    num_val = round(len(shuffled) * VAL_FRACTION)
    monitoring, baseline_train = shuffled[:num_val], shuffled[num_val:]
    monitoring_hashes = {story_hash(s) for s in monitoring}
    log(f"monitoring split reconstructed: {len(monitoring):,} stories "
        f"(baseline train {len(baseline_train):,})")

    tokenizer = BPETokenizer.load(SAVE_FILE)
    monitoring_ids = []
    for story in monitoring:
        monitoring_ids.extend(tokenizer.encode(story))
        monitoring_ids.append(tokenizer.eos_id)
    rebuilt = np.array(monitoring_ids, dtype=np.uint16)
    existing = np.load(MONITORING_TOKENS)
    monitoring_identical = bool(np.array_equal(rebuilt, existing))
    log(f"  monitoring token stream identical to stage3_val_tokens.npy: {monitoring_identical}")
    assert monitoring_identical, "monitoring split changed — aborting"

    # --- 5.6 leakage checks and removal ------------------------------------
    valid_stories = split_stories(OFFICIAL_VALID.read_text(encoding="utf-8"))
    valid_hashes = {story_hash(s) for s in valid_stories}
    baseline_train_hashes = {story_hash(s) for s in baseline_train}
    new_stories = stories[n_base:]

    overlap_baseline = sum(1 for s in new_stories if story_hash(s) in baseline_train_hashes)
    overlap_monitoring = sum(1 for s in new_stories if story_hash(s) in monitoring_hashes)
    overlap_valid = sum(1 for s in new_stories if story_hash(s) in valid_hashes)

    # Training set = every corpus story except the monitoring split and except exact
    # duplicates of monitoring or official-validation stories (decision 5).
    removed = 0
    training_stories = []
    for story in stories:
        h = story_hash(story)
        if h in monitoring_hashes:
            continue                      # the held-out split itself
        if h in valid_hashes:
            removed += 1
            continue
        training_stories.append(story)
    log(f"leakage: vs baseline-train {overlap_baseline} | vs monitoring {overlap_monitoring}"
        f" | vs official valid {overlap_valid} -> removed {removed:,} stories")

    # Near-duplicate probe: reported only, never triggers removal.
    def opening(story):
        return hashlib.sha256(" ".join(story.split())[:100].encode()).hexdigest()
    valid_openings = {opening(s) for s in valid_stories}
    monitoring_openings = {opening(s) for s in monitoring}
    near_valid = sum(1 for s in new_stories if opening(s) in valid_openings)
    near_monitoring = sum(1 for s in new_stories if opening(s) in monitoring_openings)
    log(f"near-duplicate probe (first 100 chars, reported only): "
        f"{near_valid:,} share an opening with a validation story, "
        f"{near_monitoring:,} with a monitoring story")

    manifest["leakage"] = {
        "overlap_with_baseline_train": overlap_baseline,
        "overlap_with_monitoring": overlap_monitoring,
        "overlap_with_official_valid": overlap_valid,
        "stories_removed": removed,
        "removal_policy": "exact duplicates of monitoring or official-validation stories",
        "near_duplicate_probe": {
            "method": "sha256 of first 100 whitespace-normalised characters",
            "new_stories_sharing_opening_with_official_valid": near_valid,
            "new_stories_sharing_opening_with_monitoring": near_monitoring,
            "action": "reported only, no removal",
        },
    }

    # --- 5.7 tokenizer audit ----------------------------------------------
    log("tokenising with the frozen tokenizer...")
    start = time.time()
    alphabet = set(tokenizer.alphabet)
    unseen = Counter(ch for ch in text if ch not in alphabet)

    def audit(story_list):
        tokens = unk = 0
        for story in story_list:
            ids = tokenizer.encode(story)
            tokens += len(ids) + 1           # + <EOS>
            unk += ids.count(tokenizer.unk_id)
        return tokens, unk

    corpus_tokens, corpus_unk = audit(stories)
    train_tokens, train_unk = audit(training_stories)
    baseline_train_tokens, baseline_train_unk = audit(baseline_train)
    log(f"  corpus {corpus_tokens:,} tokens | training {train_tokens:,} tokens "
        f"| {time.time() - start:.0f}s")

    corpus_chars = sum(len(s) for s in stories)
    train_chars = sum(len(s) for s in training_stories)
    baseline_chars = sum(len(s) for s in baseline_train)

    manifest["tokenizer_audit"] = {
        "tokenizer": SAVE_FILE.name, "frozen": True,
        "corpus_bpe_tokens": corpus_tokens, "corpus_unk": corpus_unk,
        "corpus_unk_rate": corpus_unk / corpus_tokens,
        "training_bpe_tokens": train_tokens, "training_unk": train_unk,
        "training_unk_rate": train_unk / train_tokens,
        "characters_outside_alphabet": {ch: n for ch, n in unseen.most_common()},
        "distinct_characters_outside_alphabet": len(unseen),
    }
    manifest["text_seen_at_budget"] = {
        "token_budget": TOKEN_BUDGET,
        "baseline": {
            "training_stories": len(baseline_train), "characters": baseline_chars,
            "tokens": baseline_train_tokens,
            "characters_per_token": baseline_chars / baseline_train_tokens,
            "characters_seen_at_budget": round(TOKEN_BUDGET * baseline_chars
                                               / baseline_train_tokens),
            "epochs_at_budget": TOKEN_BUDGET / baseline_train_tokens,
        },
        "stage6_1": {
            "training_stories": len(training_stories), "characters": train_chars,
            "tokens": train_tokens,
            "characters_per_token": train_chars / train_tokens,
            "characters_seen_at_budget": round(TOKEN_BUDGET * train_chars / train_tokens),
            "epochs_at_budget": TOKEN_BUDGET / train_tokens,
        },
    }
    manifest["environment"] = {"python": sys.version.split()[0],
                               "numpy": np.__version__}

    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"\nWrote {MANIFEST.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
