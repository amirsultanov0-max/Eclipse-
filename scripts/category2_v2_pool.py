"""
Category 2 rubric v2: build the anonymised candidate pool for worked examples.

Examples must come from training-time samples of the seed-replication runs, NOT
from the 80 scored generations. This script pools every sample in the six
logs/seed_replication/*/samples.txt files and writes two files:

  eval/category2_v2_examples/pool.json            text only, shuffled, neutral ids
  eval/category2_v2_examples/origins_WITHHELD.json  id -> run / step / source file

Candidates are chosen by reading pool.json alone. The origins file is written
but never printed; only its SHA256 is shown, so it can be revealed after the v2
package is sealed and checked against this commitment.

The shuffle uses the operating system's RNG (no seed): reproducing the order is
not needed because the origins file records it, and an unseeded shuffle means
the order cannot be reconstructed from the run order either.

It also checks that no pooled sample overlaps the 80 scored generations beyond
the shared prompt.
"""

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILES = sorted((ROOT / "logs" / "seed_replication").glob("*/samples.txt"))
GENERATIONS = ROOT / "eval" / "blind_scoring_package" / "generations.json"
OUT_DIR = ROOT / "eval" / "category2_v2_examples"
HEADER = re.compile(r"=== step (\d+) ===")
OVERLAP_WARN = 12      # words; a shared run this long beyond the prompt is reported


def split_samples(text):
    """[(step, sample_text)]. Headers can follow a sample with no newline."""
    parts = HEADER.split(text)          # ['', step, body, step, body, ...]
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def longest_shared_run(a_words, b_words):
    """Length of the longest run of consecutive words shared by a and b."""
    best, prev = 0, [0] * (len(b_words) + 1)
    for x in a_words:
        cur = [0] * (len(b_words) + 1)
        for j, y in enumerate(b_words, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def main():
    assert len(SAMPLE_FILES) == 6, f"expected 6 samples files, found {len(SAMPLE_FILES)}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_path, origins_path = OUT_DIR / "pool.json", OUT_DIR / "origins_WITHHELD.json"
    for p in (pool_path, origins_path):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p.relative_to(ROOT)}")

    entries = []
    for f in SAMPLE_FILES:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        for step, text in split_samples(f.read_text(encoding="utf-8")):
            entries.append({"text": text, "origin": {
                "run": f.parent.name, "step": step,
                "source_file": str(f.relative_to(ROOT)), "source_sha256": sha}})

    random.SystemRandom().shuffle(entries)
    for i, e in enumerate(entries, 1):
        e["id"] = f"ex_{i:02d}"

    pool = {"note": "Training-time samples, identity stripped and shuffled. "
                    "Origins are withheld until the v2 package is sealed.",
            "items": [{"id": e["id"], "text": e["text"]} for e in entries]}
    origins = {e["id"]: e["origin"] for e in entries}
    pool_path.write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    origins_path.write_text(json.dumps(origins, indent=1) + "\n", encoding="utf-8")

    # Overlap with the 80 scored generations (which must not be used as examples).
    gens = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    gens = gens.get("items", gens) if isinstance(gens, dict) else gens
    gen_words = [g["text"].split() for g in gens]
    worst = max((longest_shared_run(e["text"].split(), gw), e["id"])
                for e in entries for gw in gen_words)

    print(f"pooled {len(entries)} samples from {len(SAMPLE_FILES)} files "
          f"({sum(1 for e in entries if not e['text'])} empty)")
    print(f"exact duplicates of a scored generation: "
          f"{sum(1 for e in entries if any(e['text'] == g['text'] for g in gens))}")
    print(f"longest word run shared with any of the {len(gens)} scored generations: "
          f"{worst[0]} words ({worst[1]})"
          + ("  <- above the warning threshold, check by hand" if worst[0] >= OVERLAP_WARN else ""))
    print(f"wrote {pool_path.relative_to(ROOT)}")
    print(f"wrote {origins_path.relative_to(ROOT)} (withheld) sha256 "
          f"{hashlib.sha256(origins_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
