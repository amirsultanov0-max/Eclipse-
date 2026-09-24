# Stage 6.3 preregistration — learning-rate policy at fixed capacity, data and steps

**Preregistration date:** 2026-09-24
**Repository HEAD at drafting:** `a01f3baa735398c2ad2f5c8bb138ef55c5f6fba9` (branch `stage-6.3-lr-policy`, cut from `main`)
**Status:** draft, uncommitted. One open item (section 10) must be resolved before any treatment-arm
training.

## 1. Research question

Does linear warm-up over the first 10% of steps followed by cosine decay (peak 1e-3, final 1e-4)
improve official validation loss over a constant learning rate of 1e-3, for the 1.77M-parameter model
trained on the same 200 MB data for the same 10,000 steps?

## 2. Locked experimental design

| element | value |
|---|---|
| dataset | the exact deduplicated 200 MB corpus used for Stage 6.1 and 6.2 (`data/stage6_1_train_tokens.npy`, 49,124,477 tokens) |
| tokenizer | frozen BPE 4,000 (`tokenizer/bpe_4000.json`) |
| architecture | d_model 128, 4 heads, d_ff 512, 6 blocks, **1,767,424** parameters, both arms (the default architecture; no architecture flag is passed) |
| context length | 512 |
| batch size | 16 |
| training steps | 10,000, in one uninterrupted run per seed |
| optimizer | AdamW, weight decay 0.01 on Linear weights only, gradient clipping 1.0, as in every earlier run |
| baseline arm (reused) | constant learning rate 1e-3, no warm-up, no decay |
| treatment arm (new) | linear warm-up to 1e-3 over steps 1–1,000, then cosine decay to 1e-4 at step 10,000 (section 2.1) |
| seeds | 1337, 1338, 1339 |
| evaluation seed | 1337 |
| evaluation | same protocol and official validation files as every earlier arm (section 6) |

The only difference between the arms is the learning rate applied at each step. Throughout this
document and every result document that follows, the arms are labelled **"baseline arm (reused)"**
and **"treatment arm (new)"**.

### 2.1 The treatment schedule, exactly

Steps are numbered 1 to 10,000, as in `train_transformer.py`'s training loop. The learning rate
applied to the optimizer update of step `s` is

- for `s ≤ 1000`: `lr(s) = 1e-3 × s / 1000`
- for `s > 1000`: `lr(s) = 1e-4 + ½ × (1e-3 − 1e-4) × (1 + cos(π × (s − 1000) / 9000))`

| step | lr(s) |
|---|---|
| 1 | 1.0e-06 |
| 500 | 5.0e-04 |
| 1,000 | 1.0e-03 (peak) |
| 1,001 | 9.9999997e-04 |
| 3,250 | 8.6819805e-04 |
| 5,500 | 5.5e-04 |
| 7,750 | 2.3180195e-04 |
| 10,000 | 1.0e-04 (final) |

The schedule rises strictly through step 1,000 and falls strictly after it. **VERIFIED** (computed
from the formula above)

How it is applied:

- The learning rate is written directly into `lr` of **both** AdamW parameter groups (decayed and
  undecayed) immediately before step `s`'s optimizer update. No `torch.optim.lr_scheduler` object is
  used.
- Setting the learning rate draws no random numbers. Each seed's batch sequence, initial weights and
  evaluation batches are therefore identical in the two arms; section 10 verifies this rather than
  assuming it.
- The warm-up length (10% of the steps) and the final rate (1e-4) are fixed in the design, not
  command-line choices. The peak is `--lr`, which stays at its default of 1e-3.
- `--resume` is not allowed with the treatment schedule. Every treatment run is a single pass from
  step 1, like the baseline runs.

## 3. Arm provenance

**The baseline arm is NOT newly trained for 6.3.** It consists of the three 200 MB runs from the
seed-replication study, which Stage 6.2 also reused as its reference arm. Each was launched from a
clean tree at one commit and its results were committed in that commit's direct child:

| run | trained at (manifest `git.commit`) | results committed in | `train_transformer.py` at training |
|---|---|---|---|
| `sr_data200mb_s1337` | `0d55208` | `124f12f` | `817e03ec…` |
| `sr_data200mb_s1338` | `c970064` | `a390499` | `817e03ec…` |
| `sr_data200mb_s1339` | `f0d5dbe` | `b5cc72f` | `817e03ec…` |

**VERIFIED**: training commit and clean-tree flag from each run's `.provenance.json`, the
parent–child relation from `git log`, and the training-script hash by hashing the file at each
training commit. All three manifests record Python 3.14.7, torch 2.14.0 and numpy 2.5.3.

Only the treatment arm is newly trained.

### What was already known when this was written

The baseline arm's official validation losses were published before this preregistration, in
`eval/seed_replication_results.md` and `eval/stage6_2_preregistration.md` section 3. They are
therefore known quantities, not blind ones:

| seed | full file | clean subset | selected checkpoint |
|---|---|---|---|
| 1337 | 2.1065 | 2.1080 | step 10,000 |
| 1338 | 2.1116 | 2.1131 | step 9,750 |
| 1339 | 2.1016 | 2.1031 | step 10,000 |
| **mean** | **2.1066** | **2.1081** | |
| sample SD (n-1) | 0.0050 | 0.0050 | |
| range | 2.1016 – 2.1116 | 2.1031 – 2.1131 | |

**VERIFIED**: the losses and selected steps are copied from each run's committed
`results/seed_replication/sr_data200mb_s*_official_{full,clean}.md`; the mean, SD and range are
computed from them, and they match `eval/stage6_2_preregistration.md` section 3. Only the treatment
arm's results are unknown at preregistration time, so this is not a fully blind design.

This has a direct consequence for the verdict rule. The 3x threshold depends on the larger
within-arm SD, and the baseline arm's SD (0.0050 on both targets) is already known.

**Smallest effect that can be called ROBUST.** If the treatment arm's SD is at most 0.0050, the
larger SD is 0.0050, and a ROBUST verdict needs an effect of at least **0.0150 nats**, with
non-overlapping ranges on top of that. If the treatment arm varies more, the bar rises with it.
A real improvement smaller than this will be reported as INCONCLUSIVE, which is what the rule says,
not a finding that the schedule does not help. **INFERRED** (arithmetic from the rule and the known
SD; it matches the estimate in `eval/track_a_conclusion.md` section 11)

## 4. Compute and data exposure

Unlike Stage 6.2, this is a comparison at **fixed data exposure and fixed compute**. Both arms train
the same architecture on the same batches for the same 10,000 steps, 81,920,000 token positions per
run, and setting a learning rate costs nothing measurable per step.

What does differ is the total amount of learning rate the run receives. The treatment schedule's
mean learning rate over the 10,000 steps is 5.45e-4, **54.5%** of the baseline's constant 1e-3.
**VERIFIED** (computed from section 2.1). The comparison is between two whole policies, not between
two shapes at equal total learning rate.

Expected wall-clock time is that of the baseline runs: they took 116.5, 39.3 and 39.6 minutes (the
116.5-minute run is an outlier with no recorded cause), so about 40 minutes per treatment run, plus
about 2 minutes per official evaluation. **INFERRED**

## 5. Checkpoint selection

Identical in both arms, and unchanged from every earlier stage: the **best checkpoint is the one with
the lowest monitoring-split loss within the 10,000 steps**. The monitoring split is 50 fixed batches
drawn with evaluation seed 1337, independent of the training seed, and is evaluated every 250 steps.

**Expected asymmetry, stated in advance.** Under a constant learning rate the monitoring loss can
level off before the end: baseline seed 1338 selected step 9,750, the other two step 10,000. With the
treatment schedule the learning rate is smallest at the very end, so the treatment runs are very
likely to select step 10,000. That is the rule working as intended, not a difference in how the arms
are treated.

## 6. Primary metric

Official validation loss and perplexity, reported for **both** the full official validation file and
the clean subset. Neither is selected over the other after seeing results; both are reported
always.

Procedure, identical to the baseline arm's: `eval_official_valid.py` on each run's best checkpoint,
once for the full file and once with `--clean-subset`. The validation stream is cut into
non-overlapping 512-token windows, dropping the trailing partial window; the clean subset
additionally drops the 100 contaminated stories. The baseline arm's official evaluations were
produced with `eval_official_valid.py` `06293fa4…`. Stage 6.2 Amendment 2 reran both evaluations of
`sr_data200mb_s1337_best.pt` with the current `ccc7bb9b…` and reproduced the committed values
exactly (full file 2.1065, clean subset 2.1080; `results/equivalence_6_2/README.md`). The treatment
arm is evaluated with `ccc7bb9b…`.

## 7. Pre-registered verdict rule

"ROBUST if the two arms' ranges of official validation loss do not overlap AND the effect is at least 3x the larger within-arm SD. INCONCLUSIVE otherwise. No p-values or confidence intervals (n=3)."

Applied separately to the full validation file and to the clean subset, in the wording above, with
no other criterion.

**Sign convention:** effect = mean(baseline arm loss) - mean(treatment arm loss). A **positive**
effect means the treatment arm is better. Every effect in the result documents uses this sign, and
every perplexity change carries the sign of the loss difference it comes from. The result documents
are checked for consistency with this convention before they are committed.

No additional significance criteria will be invented after seeing results. The two targets get
separate verdicts. If the verdicts differ, both are reported and no combined verdict is formed.

| target | ranges overlap? | effect | larger within-arm SD | effect / larger SD | verdict |
|---|---|---|---|---|---|
| full validation file | to be filled | to be filled | to be filled | to be filled | to be filled |
| clean subset | to be filled | to be filled | to be filled | to be filled | to be filled |

## 8. Prediction

**Predicted: the treatment arm achieves lower official validation loss.** A gentler start should
avoid the early instability of a full-size step from random initial weights, and a decaying rate
should let the model settle into a lower loss at the end of the budget. This is a prediction, not a
guaranteed outcome.

**Interpretation condition.** The peak of 1e-3 was chosen by `lr_sweep.py` for a **constant**
schedule, and it is carried into the treatment unchanged. If the treatment arm is worse, the result
will not be reported as "warm-up and cosine decay do not help". It will be reported as **"this
schedule, at a peak tuned for a constant rate and at this budget, does not help"**. A peak chosen for
the treatment schedule could behave differently.

**Weight decay moves with the learning rate.** PyTorch's AdamW applies decoupled weight decay as
`param.mul_(1 - lr * weight_decay)` (`torch/optim/adam.py`, torch 2.14.0), so the treatment schedule
also scales the effective weight decay: it is weaker during warm-up and after the decay than under
the constant rate. **VERIFIED** (read in the installed torch 2.14.0 source). This is part of the
treatment as implemented, not a separate factor, and any result is a statement about the whole
policy.

## 9. Provenance — mandatory precondition, not an optional audit

Recorded at drafting time. The pre-training check compares against these values rather than
re-deriving them.

| item | sha256 |
|---|---|
| `train_transformer.py`, baseline arm (at `0d55208`, `c970064`, `f0d5dbe`) | `817e03ecb9bac9491c904c3ae5311540b55fc6b35dfd777f4f5b916fc3b91c76` |
| `train_transformer.py` at drafting (Stage 6.2 capacity-arm code) | `57d4e57eb7d6e6bff6c8042e2842109b0cd2f72949ef0b3ee21548842fe393a5` |
| `train_transformer.py` after Amendment 1, treatment arm | **to be recorded by Amendment 1** |
| `model/transformer.py` (both arms; unchanged since Stage 4) | `63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0` |
| `eval_official_valid.py` (treatment arm's evaluations) | `ccc7bb9bd0b98aa2d2349e58329188db17d11e3a47686d77ad9c58edd187310a` |
| `data/stage6_1_train_tokens.npy` (200 MB training stream) | `5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb` |
| `data/stage3_val_tokens.npy` (monitoring stream) | `9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e` |
| `tokenizer/bpe_4000.json` | `e3876f558896533d13b7818282c6b19b55cbf205390ad4ee9e982d105762d3df` |
| `metadata/stage6_1_corpus.json` (defines the clean subset) | `93ea3dbd08f5b421c5b3c1e9987cb891cf2d138d6ed996644b77247e9dc22d55` |

**VERIFIED**: the source-file hashes computed at drafting HEAD, and the two data hashes read from
all three baseline manifests, which agree.

**Equivalence chain for the baseline arm.** The baseline runs were trained under `817e03ec…`. Stage
6.2 Amendment 1 showed bitwise that `57d4e57e…` behaves identically at default arguments
(`results/equivalence_6_2/`). Amendment 1 here must show the same for the new file against
`57d4e57e…` (section 10). Together these link the reused baseline runs to the code the treatment arm
runs under, and justify reusing them without an unchanged file hash.

### Before any treatment-arm training

1. Verify the repository tree is clean.
2. Verify `train_transformer.py` matches the hash recorded by Amendment 1.
3. Verify `model/transformer.py` and `eval_official_valid.py` match the recorded hashes.
4. Verify the 200 MB training stream and the monitoring stream match the recorded hashes.
5. Verify the tokenizer and `metadata/stage6_1_corpus.json` match the recorded hashes.
6. Verify each run's configuration: seed, `--lr 1e-3`, `--lr-schedule warmup_cosine`, 10,000 steps,
   batch 16, the default architecture (1,767,424 parameters), no `--resume`.
7. Record the exact git commit and these hashes in each run's provenance.
8. **If any hash differs, or Amendment 1's verification has not passed, STOP.** Do not train.

## 10. OPEN ITEM — Amendment 1 is required before any treatment-arm training

`train_transformer.py` cannot run the treatment schedule, and it carries a known defect that the
record says must be fixed before Stage 6.3 trains anything (`eval/stage6_2_preregistration.md`
section 9, "Known defect in the frozen training code", and "Pending for Stage 6.3" in section 13;
`eval/track_a_conclusion.md` section 10). Both are handled in **one** amendment, with one new hash
and one verification.

### Planned change

1. **`--lr-schedule {constant, warmup_cosine}`**, default `constant`. Under `constant` no learning
   rate is ever written after the optimizer is built: the code path is the pre-change one. Under
   `warmup_cosine` the rate is set as in section 2.1. `--resume` together with `warmup_cosine` stops
   with an error.
2. **Run-summary lines generated from the run's own arguments.**
   - The architecture line is built from `--d-model`, `--n-heads`, `--d-ff`, the block count and the
     instantiated parameter count.
   - The learning-rate line states the schedule. A treatment run no longer reads as "lr 0.001".
   - At default arguments both lines produce exactly the text the current code hardcodes, so a
     baseline-style run's summary is unchanged.
   - This fixes the known defect. Already-committed summaries, including the Stage 6.2 capacity-arm
     files, are not rewritten.
3. **Recording.** The run config and provenance manifest record the schedule, the warm-up steps and
   the final rate.
   - Under `warmup_cosine` the learning rate of every step is also written to a separate per-step
     file.
   - The existing per-step log keeps its exact format, so it can still be compared bitwise at default
     arguments.

No random-number-consuming call is added, removed or reordered.

### Verification plan

It runs on the training machine: `data/` is not in the repository, and the precedent (Stage 6.2
Amendment 1) is a 1,001-step run on the real 200 MB stream. Pre-change code is `57d4e57e…`.

| check | how | pass condition |
|---|---|---|
| V1. default arguments, pre vs post | 1,001 steps, seed 1337, CPU, 200 MB stream, covering the step-1,000 in-training generation | per-step loss, per-step gradient norm, per-step batch starts, the whole per-step log file, initial-weight, eval-batch and final-weight fingerprints, and the step-1,000 sample text and draw count all **bitwise equal** |
| V2. default arguments, run summary | the V1 runs' results files | identical apart from the lines that carry the run's name, date, log-file names or wall-clock time; in particular the architecture and learning-rate lines are unchanged |
| V3. default arguments on MPS | same run on MPS | initial-weight fingerprint, eval-batch fingerprint and running batch-start hash **bitwise equal** (MPS loss is not bit-reproducible run to run, as recorded in `results/equivalence_6_2/README.md`) |
| V4. treatment schedule, same seed | 1,001 steps, seed 1337, CPU, `--lr-schedule warmup_cosine` against the V1 post-change run | batch starts, running batch-start hash, initial-weight and eval-batch fingerprints **bitwise equal** to the constant run; step-1 loss and step-1 gradient norm equal (both are computed before the first update); the per-step learning-rate file equal to section 2.1 at all 1,001 steps |
| V5. summary line at non-default arguments | a short scratch run with `--lr-schedule warmup_cosine`, and one with `--d-model 352 --n-heads 4 --d-ff 1408` | the lines state that run's actual schedule and architecture (10,537,472 parameters for the second) |

The evidence goes in `results/equivalence_6_3/` in the same form as `results/equivalence_6_2/`.

### Recording the amendment

When the change lands and V1–V5 pass, section 13 records it with what changed, why, the
verification and the new `train_transformer.py` hash. The new hash is **added** to section 9's table.
It does not replace the baseline arm's `817e03ec…` or the Stage 6.2 capacity arm's `57d4e57e…`.
`eval/stage6_2_preregistration.md` section 9 says the new hash is to be recorded there as well, as
6.3's provenance. That is an append-only note to a Phase 2 document and is made only with explicit
approval.

## 11. Stopping rule

**"Three seeds per arm are the complete experiment. No additional seed will be added after seeing
the results."**

No secondary generation evaluation is registered for 6.3. The experiment is decided on official
validation loss alone.

## 12. Interpretation boundaries

**This experiment CAN test:** whether this particular schedule (linear warm-up over 10% of the steps,
cosine decay from 1e-3 to 1e-4) improves official validation loss over a constant 1e-3, for the
1.77M-parameter model, with fixed data, fixed steps, fixed compute, and a fixed tokenizer, context
length and batch size.

**This experiment CANNOT establish:**
- anything about the 10.54M-parameter model or any other capacity; applying the schedule there
  needs its own run;
- whether any improvement comes from the warm-up, from the decay, or from the lower total learning
  rate (section 4), because all three change together;
- whether other peaks, warm-up lengths, final rates or schedule shapes would do better or worse;
- the effect of the schedule at other budgets or at convergence (Track A section 9: neither arm is
  trained to convergence at 10,000 steps);
- anything about story quality, which is not measured here.

## 13. Changes after preregistration

Any modification to the locked design requires explicit documentation **before** training, recording
what changed and why. No change may be made silently after observing results.

*No amendments yet. Amendment 1 (section 10) is to be recorded here before any treatment-arm
training.*
