# Category 2 retest: pass registry and methodology notes

## Passes

| pass | package | opening message | model | extra messages | file |
|---|---|---|---|---|---|
| 1 | v1 (eval/blind_scoring_package, sealed) | unknown | unknown | unknown | eval/blind_scores_stage6_1.json (commit 5bdbada) |
| 2 | v1 | opening_message.txt | Opus 5.5 | none | NEVER RECOVERED — the scores were reported to exist but never reached this repository. `pass2_scores.json` holds placeholder text (a stray shell command), not scores, and is untracked. |
| 3 | v1 | opening_message.txt | Opus 5.5 (per request) | not recorded | NEVER RECOVERED — same as pass 2. `pass3_scores.json` is an untracked placeholder. |
| 4 | v2 (`eval/blind_scoring_package_v2`, sealed, commit `c80ad6a`) | opening_message.txt | Opus 5.5 | none | `pass4_scores.json` — copied byte-for-byte from the scorer's output, sha256 `d196f8e1d293edc4…` |
| 5 | v2, same sealed package | opening_message.txt | Opus 5.5 | none | `pass5_scores.json` — copied byte-for-byte, sha256 `5cf48272d3ab4d5f…` |

Passes 4 and 5 were each scored in a fresh incognito chat, with the package hashes verified before
scoring. Both validated on arrival: exactly 80 entries, ids exactly the sealed map's with none
missing, duplicated or extra, all four flags boolean, and no example ids from `examples.md` present.

Passes 1-3 used the v1 package and are not relabelled or modified. All passes score the same
80-generation test set.

## Known gap — the v1 test-retest pair was never measured

Passes 2 and 3 were run against the v1 package but their score files were never recovered, so the
v1 test-retest reliability was never computed. The planned rubric-change diagnostics (pass 4 vs
pass 2, pass 5 vs pass 3) are therefore **not possible** and were skipped, not deferred: there is no
v1 pair to compare against, and no v1 reliability figure exists for any comparison.

What survives: pass 1 (`eval/blind_scores_stage6_1.json`, commit `5bdbada`), whose opening message
and model are unknown, and the v2 pair below. A v1-to-v2 comparison against pass 1 alone would
confound the rubric change with an unknown prompt and an unknown model, so none is reported.

## Pass 4 vs pass 5 — test-retest reliability on the sealed v2 package

Computed without opening the sealed map's values; only its ids were read.

| item | pass 4 flags | pass 5 flags | agreement | Cohen's kappa |
|---|---|---|---|---|
| unexplained_object | 45/80 | 48/80 | 93.8% (75/80) | 0.87 |
| uncaused_action | 71/80 | 69/80 | 95.0% (76/80) | 0.77 |
| character_discontinuity | 40/80 | 35/80 | 93.8% (75/80) | 0.88 |
| contradicted_ending | 7/80 | 4/80 | 93.8% (75/80) | 0.51 |

- overall agreement: 301/320 judgments (94.1%)
- generations scored identically on all four items: 64/80
- mean flags per generation: pass 4 2.04, pass 5 1.95

**VERIFIED** (computed from `pass4_scores.json` and `pass5_scores.json`).

Kappa depends on the base rate as well as on agreement: `contradicted_ending` is flagged 7 and 4
times out of 80, so its 93.8% agreement yields kappa 0.51, while `character_discontinuity` reaches
0.88 at the same agreement because it is flagged far more often. The two figures are not
interchangeable, and the rarest item is the least reliably measured.

## Methodology note (v1 -> v2)

v2 changed two things relative to v1: the character_discontinuity definition and the
addition of worked examples containing full four-item scores. Therefore v1->v2 differences
in ANY checklist item are rubric-change diagnostics and cannot be interpreted as pure scorer
or model variation.

## Post-pool decision

"gender" was added to clause (a) of the v2 character_discontinuity definition after the
candidate pool had been seen, to resolve gender-flip cases the definition otherwise left
ambiguous. Recorded with its reason in eval/category2_v2_examples/package_notes.md.
