# Stage 6.1 — results: more training data at a fixed compute budget

_2026-09-22. Follows the reporting plan pre-registered in `docs/stage6_1_design.md`
(§9.1–§9.6). No new experiments; every number here comes from measurements already
taken._

## 1. Summary

Holding the compute budget, architecture, tokenizer and optimization procedure fixed
while increasing unique training data from 4.69M to 49.12M tokens improved official
validation perplexity from **9.20 to 8.25** on the full file and from **9.23 to 8.26**
on the contamination-free clean subset, a reduction of about 10% either way. The
train/validation gap, which reached **+0.243** for the baseline over 17.5 passes,
stayed within **±0.013** for 6.1 over 1.67 passes. On the 40-generation structured
evaluation, blind-scored in a fresh session, mean causal flags per generation moved
from **2.33 to 2.08** and name retention among named generations from **52% to 65%**,
while grammaticality was unchanged at 27/40 and generations were about 11% shorter.
The cross-session drift measured on the baseline's own 40 generations (**3.33 → 2.33**
mean flags) is four times larger than the difference between the two models, so the
category-2 comparison carries much less weight than the perplexity result.

## 2. What was held fixed, and what changed

**Fixed** (confirmed identical, design §4): architecture (1,767,424 params, T=512),
tokenizer `bpe_4000.json` frozen and not retrained, AdamW at lr 1e-3 with the same
weight-decay grouping, gradient clipping at 1.0, 10,000 steps at batch 16
(81,920,000 tokens), seed 1337, the same 1,109-story monitoring split (verified
byte-identical to `stage3_val_tokens.npy`), and the frozen Stage 5 harness.

**Changed**: the training corpus.

| | Baseline | Stage 6.1 |
|---|---|---|
| Corpus bytes | 19,999,022 | 199,999,176 |
| Training stories | 21,065 | 217,080 |
| Training tokens | 4,691,267 | 49,124,477 |
| Passes over the data at the fixed budget | 17.46 | **1.67** |
| Characters seen at the budget | 325,389,095 | 324,701,129 |
| Characters per token | 3.9720 | 3.9636 |
| `<UNK>` rate under the frozen tokenizer | — | 2.2e-06 (109 tokens) |

The frozen tokenizer cost 6.1 only **0.21% less text** for the same token budget, so
the compression-drift confound flagged in the design is negligible in size.

**Deviation from the baseline's procedure** (design §8): 6.1 ran as a single
continuous 10,000-step run, where the baseline ran 5,000 steps plus a resumed 5,000.
Optimizer semantics are identical; the difference is limited to which random windows
were drawn.

## 3. Official validation (pre-registered, §9.4)

Full deterministic pass, non-overlapping 512-token windows, every token scored once.

| Model (step 9,750) | Full file | Clean subset |
|---|---|---|
| Baseline | 2.2197 / **9.20** | 2.2223 / **9.23** |
| **Stage 6.1** | **2.1097 / 8.25** | **2.1110 / 8.26** |
| Difference | −0.1100 nats (−10.4%) | −0.1113 nats (−10.5%) |

Percentage changes are computed from the loss difference (`exp(Δloss) − 1`), not from
the rounded perplexities, which would give −10.3% and −10.5%.

Full file: 21,990 stories, 4,787,559 tokens scored. Clean subset: 21,890 stories
(100 excluded by the hashes sealed in `metadata/stage6_1_corpus.json` before
training), 4,756,899 tokens.

**The contamination correction turned out to be immaterial.** Removing the 92 stories
the baseline had trained on moved its loss by 0.0026 nats. The improvement is
essentially the same on both targets, so the result does not depend on the
contamination question that prompted the clean subset.

## 4. Training

### 4.1 Monitoring-split curve at matching steps

| Step | Baseline val | 6.1 val | Diff |
|---|---|---|---|
| 250 | 4.0733 | 4.0492 | −0.0242 |
| 1,000 | 3.0068 | 2.9904 | −0.0164 |
| 2,000 | 2.5939 | 2.5441 | −0.0497 |
| 3,000 | 2.4379 | 2.3757 | −0.0623 |
| 5,000 | 2.3079 | 2.2237 | −0.0842 |
| 7,000 | 2.2481 | 2.1480 | −0.1001 |
| 9,000 | 2.2210 | 2.1030 | −0.1180 |
| 10,000 | 2.2074 | 2.0875 | −0.1199 |

6.1 leads at every checkpoint, and the margin widens monotonically.

### 4.2 Best checkpoint

Both runs selected step **9,750** under the same policy (best monitoring-split loss).

| | Baseline | 6.1 |
|---|---|---|
| Monitoring loss | 2.2031 | **2.0868** |
| Monitoring perplexity | 9.053 | **8.059** |

### 4.3 Train/validation gap

| Step | Baseline | 6.1 |
|---|---|---|
| 3,000 | +0.1043 | −0.0146 |
| 5,000 | +0.1589 | −0.0050 |
| 7,000 | +0.1971 | −0.0031 |
| 10,000 | **+0.2432** | **+0.0120** |

This is the clearest structural difference between the runs. At 1.67 passes a training
batch is close to held-out data, so there is little opportunity to fit the training
stream specifically; at 17.5 passes there is considerably more.

## 5. Gradient-norm audit

| | Baseline (10k) | 6.1 |
|---|---|---|
| Median | 0.756 | 0.722 |
| Max | 2.198 (2.9× median) | **45.742 (63.3× median)** |
| Steps clipped | 0.4% | 0.4% |
| **Steps above 10× median** | 0 | **1 (step 17)** |

**Flagged per the committed rule.** One isolated spike at step 17, with steps 15, 16,
18 and 19 at 1.21, 1.18, 1.06 and 1.08. Excluding the first 100 steps, the run is
calmer than the baseline: max 2.36, 3.3× median.

**Diagnostic (reported only; nothing was changed as a result).** The step-17 batch was
reproduced from the fixed seed — replayed norm 45.744 against the logged 45.742 — and
inspected. It contains **zero `<UNK>` tokens**, 34 `<EOS>` tokens (ordinary for 16
windows of 512), and ordinary corpus prose. Training loss rose slightly at that step
(6.5629 → 6.6492). No data anomaly was found; on this evidence it looks like an early
optimization transient that clipping absorbed, and the loss curve shows no disruption
(the step-250 evaluation is 4.0492, already ahead of the baseline's 4.0733).

## 6. Mechanical metrics (categories 1 and 3)

Frozen harness: the 8 prompts in `eval/prompts.json`, top-k 40, seeds 1337–1341,
150-token cap, 40 generations per model.

| Metric | Baseline | 6.1 |
|---|---|---|
| **Name retention (of named generations)** | 13/25 = **52%** | 13/20 = **65%** |
| — recurs / changes / disappears | 13 / 5 / 7 | 13 / 4 / 3 |
| — no proper noun at all | 15/40 | 20/40 |
| Uninitiated new name | 12/40 | 12/40 |
| Reached `<EOS>` | 35/40 | 32/40 |
| Ends on terminal punctuation | 36/40 | 32/40 |
| Has invented words | 9/40 | 7/40 |
| **Grammatical** | **27/40** | **27/40** |
| Mean 4-gram repetition | 2.0% | 1.4% |
| **Mean length (tokens)** | **102.8** | **91.0** |
| Median length | 108 | 90 |

**Length check (§9.6), applied to the name-retention figure.** The numerator is
identical in both models — 13 generations in which the first name recurs. What changed
is the denominator: 6.1 produced 20 named generations against the baseline's 25,
because 20 of its 40 generations contain no proper noun at all. The rise from 52% to
65% is therefore a denominator effect, and it sits alongside an 11% fall in mean
length, which gives fewer opportunities to introduce or drop a name. The evidence here
is consistent with shorter output rather than with demonstrably better name tracking.

**On the invented-word rule.** The committed rule judges against
`data/tinystories_train_subset.txt`, the baseline's corpus, for both models. Checked
against the full 200 MB corpus instead, the counts are unchanged (9/40 and 7/40):
every flagged word (`dading`, `jplipted`, `ruinarian`, `shookterrautions`) is absent
from both corpora, so the fixed reference does not distort the comparison.

## 7. Category 2 — causal consistency (blind, fresh session)

**Scorer provenance.** The 80 generations were scored in a **fresh incognito
claude.ai chat that saw only the four files of the scoring package** — README,
checklist, the 80 shuffled unlabeled generations, and the output format. It was not
Claude Code, had no history of this project, and had no access to the repository. The
sealed mapping (`eval/blind_map_stage6_1.json`) was committed before scoring; scores
arrived as `eval/blind_scores_stage6_1.json` (80 entries, validated as an exact ID
match against the map). Neither file was modified during unblinding.

| Checklist item | Baseline | 6.1 | Difference |
|---|---|---|---|
| unexplained_object | 24/40 (60%) | 24/40 (60%) | 0 |
| uncaused_action | 37/40 (93%) | 33/40 (83%) | −4 |
| character_discontinuity | 25/40 (63%) | 21/40 (53%) | −4 |
| contradicted_ending | 7/40 (18%) | 5/40 (13%) | −2 |
| **Mean flags per generation** | **2.33** | **2.08** | **−0.25** |
| At least one flag | 39/40 | 36/40 | −3 |
| Zero flags | 1/40 | 4/40 | +3 |

**"At least one flag" was pre-declared saturated and is not a success criterion**
(§9.2). It was 40/40 in the original Stage 5.5 scoring; under this scorer it is 39/40
and 36/40, so it is not fully saturated here, but it remains reported only.

**Supporting detail.** Paired by prompt and seed, 6.1 has fewer flags in 16 pairs,
more in 10, and the same in 14. Generations with zero flags are markedly shorter than
flagged ones (baseline: 18 tokens, n=1; 6.1: 47 tokens, n=4; flagged generations
average 105 and 96 tokens), so the same length caveat that applies to name retention
applies here: part of the reduction is consistent with shorter output.

## 8. Cross-session scorer drift

The baseline's 40 generations were scored twice: once in Stage 5.5 by Claude Code with
full project context, and once now by the fresh session. This cross-session comparison
is a reliability/drift observation only, and it is confounded by differences in scoring
context: Stage 5.5 was scored non-blind inside the project session, while the fresh
session scored a blind, frozen package; the Stage 5.5 conventions were written down
during scoring, while the fresh session received them up front; and the Stage 5.5
instructions cannot be recovered. The 1.00-flag difference is therefore not a measured
scorer-noise magnitude. It is not used to compare the two models, which are compared
only within the single blind pass in §7, where both were scored in the same pass by the
same scorer.

**Scoring instructions compared (2026-09-22): not byte-identical.** The four checklist definitions are word-for-word identical after whitespace collapse (`eval_error_analysis.py` @ `2f5d7dd` vs `eval/blind_scoring_package/checklist.md` @ `84839a1`), but the Stage 6.1 package reformatted them, reworded the preamble, supplied the conventions as up-front instructions (in Stage 5.5 they were recorded by the scorer alongside its scores, in `eval/causal_scores.json`) with one added sentence, and added a README and output format; the Stage 5.5 scorer's full instruction context (the Claude Code session) and the message typed into the fresh claude.ai chat are not stored in the repository and could not be compared.

| Checklist item | Stage 5.5 | Fresh session | Agreement |
|---|---|---|---|
| unexplained_object | 31/40 | 24/40 | 33/40 |
| uncaused_action | 40/40 | 37/40 | 37/40 |
| character_discontinuity | 38/40 | 25/40 | 25/40 |
| contradicted_ending | 24/40 | 7/40 | 23/40 |
| **Mean flags per generation** | **3.33** | **2.33** | — |

Per-flag agreement across all 160 judgments is **118/160 (73.8%)**, and only **11 of
40** generations were scored identically on all four items. The fresh session applied
the checklist more conservatively throughout, most sharply on `contradicted_ending`
(24 → 7; all 7 of its flags fall inside the original 24) and `character_discontinuity`
(38 → 25).

**This drift is four times the size of the model difference it is being used to
measure** (1.00 versus 0.25 mean flags). The comparison in §7 remains internally
valid, because both models were scored in the same pass by the same scorer under the
same conditions — which is precisely why the design required the baseline to be
re-scored rather than compared against its Stage 5.5 numbers. But the absolute flag
rates are clearly scorer-dependent, and a −0.25 difference measured on an instrument
that moves by 1.00 between sessions should be treated as weak evidence.

## 9. Notes carried from the pre-registered plan (§9.5)

1. **Duplicates inside the training corpus were kept.** The 200 MB corpus contains
   2,950 exact-duplicate story texts. They were left in place to match the baseline's
   condition; removing them from 6.1 alone would have introduced a second asymmetry.
2. **Near-duplicate openings were reported, not removed.** 11,377 new stories share a
   first-100-character opening with an official validation story, and 3,366 with a
   monitoring story. TinyStories openings are formulaic, so a shared opening is weak
   evidence of a duplicated story; this remains an unquantified residual leakage risk.
3. **The Stage 4/5 official perplexity of 9.20 was measured on a validation file
   containing 92 stories the baseline trained on.** That caveat applies to every prior
   official figure, including 10.21 at 5k. The clean-subset measurement now quantifies
   it: 9.23 instead of 9.20, a 0.0026-nat effect.

Additionally, 742 stories that exactly duplicate an official validation story were
removed from the 6.1 training corpus (92 from the baseline portion, 650 from the new
material), and 64 new stories duplicating a monitoring story were excluded. The 6.1
training set is therefore not a strict superset of the baseline's.

## 10. Limitations

1. **The intervention is not a single microscopic variable.** At a fixed token budget,
   "more unique data" and "fewer repetitions per example" cannot be separated; 17.46
   passes became 1.67. Both are part of what changed.
2. **Category 2 rests on a single automated scorer**, one fresh session applying the
   checklist without human validation. The §8 drift figure measures cross-session
   consistency between two applications by the same model class, not inter-rater
   agreement among independent scorers, and not correctness.
3. **No human-written baseline was scored.** The checklist has never been applied to
   real TinyStories text, so the absolute flag rates still have no reference point.
4. **Small samples.** 40 generations per model; 25 and 20 of them named. One
   generation is 2.5 percentage points of any 40-denominator rate, so the −4/40 item
   shifts in §7 are four generations each. No pre-registered thresholds were set
   (§9.3), and these magnitudes are within the range that sampling noise could produce.
5. **Single seed throughout.** One training seed per model, five generation seeds. No
   seed sweep was run, so run-to-run variance for this setup is unmeasured and it is
   not known how the 0.11-nat perplexity difference compares to it. The consistency of
   the monitoring-split lead — present at every one of 40 evaluations and widening
   monotonically — is supporting evidence from within the single run, not a substitute
   for replication.
6. **Top-k 40 is an evaluation configuration**, not an established best decoding
   strategy.
7. **Prompt coverage.** All 8 prompts are TinyStories-style story openings.
8. **Category 1's detector has known false positives.** The uninitiated-name rate is
   reported at 12/40 for both models. In each model, 4 of those 12 generations are
   flagged only on sentence-initial words the exclusion list does not cover (baseline:
   `Whatever`, `Inside`, `Of`, `Welcome`; 6.1: `Not`, `Before`, `Inside`, `Yay`),
   leaving 8/40 with a genuine uninitiated name in both. The rule was fixed before
   scoring and left unchanged; the correction does not alter the comparison, since it
   is the same in both models.

## 11. Conclusion

Increasing unique training data by roughly 10× at a fixed compute budget produced a
clear improvement in language modelling — official validation perplexity fell from
9.20 to 8.25 on the full file and from 9.23 to 8.26 on the clean subset, about 10% in
both cases — and it visibly changed the training dynamics, with the train/validation
gap staying near zero instead of opening to +0.243. On the structured evaluation, the
entity and causal measures moved in the same direction but by much smaller amounts:
mean causal flags fell from 2.33 to 2.08, name retention among named generations rose
from 52% to 65% on an unchanged numerator of 13, and grammaticality was identical at
27/40. Both of those shifts coincide with an 11% fall in generation length, and the
causal measure sits on an instrument whose cross-session drift (1.00 mean flags) is
four times the effect being measured (0.25). Within this evaluation, and from a single
training seed per model, more data at fixed compute is clearly supported as an
improvement to next-token prediction, and only weakly supported as an improvement to
entity and causal consistency. Greater capacity and a different context length remain
the other two interventions in the Stage 6 question.
