"""
Stage 6.1: build the blind category-2 scoring package.

Implements docs/stage6_1_design.md section 10, steps 2-5. It combines the two
40-generation sets, shuffles them with a fixed seed, strips every label, writes
the sealed mapping to a separate file, and assembles a self-contained package
that a fresh session can score with no knowledge of this project.

This script does NOT score anything.

    venv/bin/python blind_rescore.py
"""

import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("baseline", PROJECT_ROOT / "eval" / "generations_topk40.json"),
    ("stage6_1", PROJECT_ROOT / "eval" / "generations_topk40_6_1.json"),
]
BLIND_SET = PROJECT_ROOT / "eval" / "blind_set_stage6_1.json"
SEALED_MAP = PROJECT_ROOT / "eval" / "blind_map_stage6_1.json"
PACKAGE = PROJECT_ROOT / "eval" / "blind_scoring_package"
SHUFFLE_SEED = 1337

CHECKLIST = """# Category 2 checklist

Four independent yes/no questions per generation. Apply the definitions exactly as
written. Do not write free-form commentary; each "yes" records the specific phrase in
the text that triggered it.

## 2a. unexplained_object

A concrete object is referred to with a definite article or possessive ("the box",
"his kite") when that object has not been mentioned earlier in the prompt or
generation. Generic scenery tied to a location already named (e.g. "the grass" after
"the park") does NOT count.

## 2b. uncaused_action

At least one stated action or emotional state cannot be traced to anything in the
preceding sentences — e.g. a character becomes sad with no preceding event, or
performs an action whose precondition never occurred.

## 2c. character_discontinuity

A character enters after the opening with no introduction, OR a character established
in the scene stops being referenced while the scene continues, with no exit stated.

## 2d. contradicted_ending

The final sentence asserts a state inconsistent with something stated earlier (e.g.
everyone is happy though the stated problem was never resolved; an object described as
lost or broken is used intact).

# Conventions

- Animals and people count as characters (2c); inanimate things count as objects (2a).
- Outdoor scenery (trees, grass, sky, bushes) is treated as generic and is not flagged
  under 2a.
- A generation that was cut off by the length cap has no ending, so
  `contradicted_ending` is false for it by default. The item records whether it was
  cut off (`truncated: true`).
- Score every item independently. Judge only the text in front of you.
"""

README = """# Scoring task

This package contains 80 short story continuations produced by small language models
trained on a children's-story corpus. Each item has a prompt and the text the model
generated from it.

Your task: score **every** item against the four yes/no questions in `checklist.md`,
and return one JSON object in the format shown in `output_format.json`.

- `generations.json` holds the 80 items. Each has an `id`, a `prompt`, a `text`, and a
  `truncated` flag.
- Score each item on its own, in the order given. Apply the checklist definitions
  literally; do not adjust the bar between items.
- For every flag you set to `true`, add a short quoted phrase from the text to the
  `evidence` object naming what triggered it. For flags set to `false`, add nothing.
- Return scores for all 80 ids. Do not omit any, and do not add ids that are not in
  `generations.json`.

The items are in random order and carry no information about which model produced
them. That is deliberate: please do not attempt to infer or group them by source.
"""

OUTPUT_FORMAT = {
    "_comment": ("Return exactly this structure with one entry per id in "
                 "generations.json. The two entries below are a FORMAT EXAMPLE ONLY — "
                 "the booleans are illustrative, not answers."),
    "scores": {
        "gen_0001": {
            "unexplained_object": True, "uncaused_action": True,
            "character_discontinuity": False, "contradicted_ending": False,
            "evidence": {"unexplained_object": "the red bucket",
                         "uncaused_action": "she was suddenly afraid"},
        },
        "gen_0002": {
            "unexplained_object": False, "uncaused_action": False,
            "character_discontinuity": False, "contradicted_ending": False,
            "evidence": {},
        },
    },
}


def main():
    items, mapping = [], {}
    pool = []
    for source, path in SOURCES:
        for record in json.loads(path.read_text(encoding="utf-8")):
            pool.append((source, record))

    random.Random(SHUFFLE_SEED).shuffle(pool)

    for index, (source, record) in enumerate(pool, start=1):
        blind_id = f"gen_{index:04d}"
        items.append({"id": blind_id, "prompt": record["prompt"], "text": record["text"],
                      "truncated": not record["reached_eos"]})
        mapping[blind_id] = {"source_model": source, "original_id": record["id"],
                             "prompt_id": record["prompt_id"], "seed": record["seed"]}

    blind_payload = {"items": items}
    BLIND_SET.write_text(json.dumps(blind_payload, indent=1, ensure_ascii=False),
                         encoding="utf-8")
    blind_sha = hashlib.sha256(BLIND_SET.read_bytes()).hexdigest()

    SEALED_MAP.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Sealed before any category-2 score exists. Unblinds "
                 "eval/blind_causal_scores_stage6_1.json after a fresh session returns it."),
        "shuffle_seed": SHUFFLE_SEED,
        "sources": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in SOURCES},
        "blind_set_sha256": blind_sha,
        "counts": {name: sum(1 for v in mapping.values() if v["source_model"] == name)
                   for name, _ in SOURCES},
        "mapping": mapping,
    }, indent=1), encoding="utf-8")

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    (PACKAGE / "README.md").write_text(README, encoding="utf-8")
    (PACKAGE / "checklist.md").write_text(CHECKLIST, encoding="utf-8")
    (PACKAGE / "generations.json").write_text(
        json.dumps(blind_payload, indent=1, ensure_ascii=False), encoding="utf-8")
    (PACKAGE / "output_format.json").write_text(
        json.dumps(OUTPUT_FORMAT, indent=1), encoding="utf-8")

    print(f"blind set: {len(items)} items -> {BLIND_SET.relative_to(PROJECT_ROOT)}")
    print(f"  sha256 {blind_sha}")
    print(f"  shuffled with seed {SHUFFLE_SEED}; "
          f"counts {json.loads(SEALED_MAP.read_text())['counts']}")
    print(f"sealed map -> {SEALED_MAP.relative_to(PROJECT_ROOT)}")
    print(f"package    -> {PACKAGE.relative_to(PROJECT_ROOT)}/ "
          f"({', '.join(sorted(p.name for p in PACKAGE.iterdir()))})")
    leaked = [k for k in items[0] if k not in {"id", "prompt", "text", "truncated"}]
    print(f"label leakage check: item keys {sorted(items[0])} | unexpected {leaked or 'none'}")


if __name__ == "__main__":
    main()
