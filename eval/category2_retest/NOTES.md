# Category 2 retest: pass registry and methodology notes

## Passes

| pass | package | opening message | model | extra messages | file |
|---|---|---|---|---|---|
| 1 | v1 (eval/blind_scoring_package, sealed) | unknown | unknown | unknown | eval/blind_scores_stage6_1.json (commit 5bdbada) |
| 2 | v1 | opening_message.txt | Opus 5.5 | none | pass2_scores.json (pending: file currently holds placeholder text, not scores). Chat printed its JSON twice; the user reports the copies were identical. |
| 3 | v1 | opening_message.txt | Opus 5.5 (per request) | not yet recorded | pass3_scores.json (pending: placeholder text) |
| 4 | v2 (eval/blind_scoring_package_v2, once sealed) | tbd | tbd | tbd | pass4_scores.json (pending) |
| 5 | v2 | tbd | tbd | tbd | pass5_scores.json (pending) |

Passes 1-3 used the v1 package and are not relabelled or modified. All passes score the same
80-generation test set.

## Methodology note (v1 -> v2)

v2 changed two things relative to v1: the character_discontinuity definition and the
addition of worked examples containing full four-item scores. Therefore v1->v2 differences
in ANY checklist item are rubric-change diagnostics and cannot be interpreted as pure scorer
or model variation.

## Post-pool decision

"gender" was added to clause (a) of the v2 character_discontinuity definition after the
candidate pool had been seen, to resolve gender-flip cases the definition otherwise left
ambiguous. Recorded with its reason in eval/category2_v2_examples/package_notes.md.
