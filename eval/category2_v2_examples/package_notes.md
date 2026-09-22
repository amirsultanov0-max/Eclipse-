# Category 2 scoring package v2 — package notes

Provenance and decisions for `eval/blind_scoring_package_v2/`. NOT part of the package:
scorers never see this file. Status: **sealed 2026-09-22** (version v2; hashes below).

## What changed from v1

| File | Change |
|---|---|
| `checklist.md` | Section 2c replaced with the v2 `character_discontinuity` definition (clauses (a), (b), (c) plus an explicit do-not-flag list). Sections 2a, 2b, 2d and Conventions are byte-identical to v1. |
| `examples.md` | New: eight worked examples, each with all four scores and a one-line reason per score. |
| `README.md` | One bullet added, pointing at `examples.md` and stating the examples are not to be scored. |
| `generations.json`, `output_format.json` | Copied from v1, byte-identical (same 80 items, same output format). |

## Decisions

- **"gender" added to clause (a)'s attribute list** — decided by the user AFTER seeing the
  anonymised candidate pool, not before. Reason: samples in the pool flip a character's
  gender ("a little girl named Samie ... He shouted"), which v2 otherwise left ambiguous
  between an attribute contradiction and the do-not-flag item "pronouns that are merely
  unclear". Clause (a) now reads: name, species, gender, age, family role, or physical property.
- **No positive example of clause (c)** — none of the 60 candidate samples contains a clear
  case of a character stated to have left, be absent or be asleep, then acting as present
  without a stated return. Clause (c) stays in the definition; `examples.md` says no example
  of it is included. One example (a "went to bed early" near-miss) shows where (c) does NOT apply.
- **No positive example of `contradicted_ending`** — all eight examples hit the 60-token
  generation cap, so `truncated` is true and, by the Conventions, `contradicted_ending` is
  false by default in every example.
- **`unexplained_object` set to false in Examples 5 and 8** ("the roof", "her house") — decided
  by the user at review. Reason: both readings are defensible, and an instructional example must
  not pick a side where a scorer could reasonably read it either way; flagging a possessive for
  the protagonist's own house would teach exactly the over-flagging v2 aims to remove.
- **Intended scores** were drafted by the assistant and reviewed by the user before sealing.

## Known limitation

The package contains **no positive `contradicted_ending` example**: no sample in the candidate
pool ended naturally, since every one hit the 60-token generation cap, so `truncated` is true
and the Conventions make `contradicted_ending` false by default in all eight. The gap was
accepted rather than generating new samples. `examples.md` states this, so the absence is not
read as a base rate.

## Example sources

Candidates came only from training-time samples of the six seed-replication runs, pooled,
identity-stripped and shuffled by `scripts/category2_v2_pool.py`; none is among the 80
scored generations (longest shared word run: 5). The user selected the eight from the
anonymised pool without seeing run, config or seed.

| package example | pool id |
|---|---|
| Example 1 | ex_43 |
| Example 2 | ex_46 |
| Example 3 | ex_33 |
| Example 4 | ex_25 |
| Example 5 | ex_55 |
| Example 6 | ex_44 |
| Example 7 | ex_32 |
| Example 8 | ex_12 |

Rejected during selection: ex_37 (multiple violations), ex_35 (mixes a violation and a
non-violation), ex_57 (sits on the (b) boundary), ex_38 (ex_55 is cleaner).

Origins (run, seed, step per pool id) are withheld in `origins_WITHHELD.json`,
sha256 `f68b87a92a2be637e2dbf36539176d724adff4995af10ab1f6b027b404d6aa8f`, recorded before
selection. They are to be revealed here only after the package is sealed.

## Sealing

Version **v2**, sealed 2026-09-22. Repository HEAD at sealing: `1c6cf10`. The package files are committed in `c80ad6a`, which is the commit to cite for the package contents.

| file | sha256 |
|---|---|
| `README.md` | `10bdfc20b4b2371f3e0f5e34ae4580bb16e7ba5380bf4edc308bb52113d18081` |
| `checklist.md` | `e945ee36aa9ef945594b8b46642973f3649a732e9ff0e687854fde49be1fa1bd` |
| `examples.md` | `e70e4ec3010cb1705e8c0e53eb26148c59f178720ff3a201422aec8514524a35` |
| `generations.json` | `db1228c93345ebc94a84b4285161651d2c3e41f23b18e3ed0a2ac13af01ecc8e` |
| `output_format.json` | `86575673182a788404619d647f0728b74bf042fe07163c7e18ed27a02fd93bc3` |

`generations.json` and `output_format.json` carry the same hashes as v1 (`db1228c93345ebc9…`, `86575673182a7884…`), so the 80 items and the output format are provably unchanged between packages.

## Example origins (revealed after sealing)

The withheld file matched its pre-selection commitment (`f68b87a92a2be637…`) when opened.

| package example | pool id | run | step |
|---|---|---|---|
| Example 1 | ex_43 | sr_data200mb_s1337 | 5,000 |
| Example 2 | ex_46 | sr_base20mb_s1339 | 6,000 |
| Example 3 | ex_33 | sr_base20mb_s1338 | 5,000 |
| Example 4 | ex_25 | sr_base20mb_s1338 | 3,000 |
| Example 5 | ex_55 | sr_data200mb_s1338 | 7,000 |
| Example 6 | ex_44 | sr_data200mb_s1339 | 5,000 |
| Example 7 | ex_32 | sr_base20mb_s1338 | 8,000 |
| Example 8 | ex_12 | sr_data200mb_s1337 | 6,000 |

