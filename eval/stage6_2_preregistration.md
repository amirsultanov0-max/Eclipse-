# Stage 6.2 preregistration — capacity at fixed data exposure

**Preregistration date:** 2026-09-22
**Repository HEAD at drafting:** `842cbcb73d3dd5183e2be5b1b3026c5480728615`
**Status:** draft, uncommitted. One open item (section 10) must be resolved before any training.

## 1. Research question

Does increasing model capacity from 1.77M to approximately 10.5M parameters improve
language-model performance under the same 200 MB data and 10,000-step training budget?

## 2. Locked experimental design

| element | value |
|---|---|
| dataset | the exact deduplicated 200 MB corpus used for Stage 6.1 (`data/stage6_1_train_tokens.npy`, 49,124,477 tokens) |
| tokenizer | frozen BPE 4,000 (`tokenizer/bpe_4000.json`) |
| context length | 512 |
| batch size | 16 |
| training steps | 10,000 |
| reference arm (reused) | d_model 128, 4 heads, d_ff 512, 6 blocks, **1,767,424** parameters |
| capacity arm (new) | d_model 352, 4 heads, d_ff 1408, 6 blocks, **10,537,472** parameters |
| optimizer | same AdamW configuration as the established 200 MB reference runs: weight decay 0.01 on Linear weights only, gradient clipping 1.0 |
| learning rate | peak LR 1e-3, LR policy identical to the reference arm — constant, no warm-up and no decay |
| seeds | 1337, 1338, 1339 |
| evaluation seed | 1337 |
| evaluation | same protocol and official validation files as the reference arm (section 6) |

Parameter counts are exact, taken from instantiated models. **VERIFIED**

## 3. Arm provenance

**The reference arm is NOT newly trained for 6.2.** It consists of three existing runs from the
seed replication study, trained 2026-09-22 under the same protocol with full provenance:

| run | commit | seed |
|---|---|---|
| `sr_data200mb_s1337` | `124f12f` | 1337 |
| `sr_data200mb_s1338` | `a390499` | 1338 |
| `sr_data200mb_s1339` | `b5cc72f` | 1339 |

All three commits and their run names were confirmed present in this repository. **VERIFIED**

Only the d352 arm is newly trained. Throughout this document and every result document that
follows, the arms are labelled **"reference arm (reused)"** and **"capacity arm (new)"**.

### What was already known when this was written

The reference arm's official validation losses were published before this preregistration, in
`eval/seed_replication_results.md`. They are therefore known quantities, not blind ones:

| seed | full file | clean subset |
|---|---|---|
| 1337 | 2.1065 | 2.1080 |
| 1338 | 2.1116 | 2.1131 |
| 1339 | 2.1016 | 2.1031 |
| **mean** | **2.1066** | **2.1081** |
| sample SD (n-1) | 0.0050 | 0.0050 |
| range | 2.1016 – 2.1116 | 2.1031 – 2.1131 |

**VERIFIED** (values copied from `eval/seed_replication_results.md` §2; mean, SD and range computed
from them). Only the capacity arm's results are unknown at preregistration time. This asymmetry is
inherent to reusing runs and is recorded here so it is not mistaken for a fully blind design.

This has a direct consequence for the verdict rule: the 3x threshold depends on the larger
within-arm SD, and the reference arm's SD (0.0050 on both targets) was known at preregistration
time; only the capacity arm's SD was unknown.

## 4. Compute versus data exposure

This is **not** a fixed-compute experiment.

> Both arms receive the same 81,920,000 token positions. The larger arm uses approximately 2.5x the
> compute per step (measured: 422 ms versus 166 ms), so training compute is intentionally not held
> constant. This experiment tests capacity at fixed data exposure, not at fixed compute. A
> fixed-compute comparison would be a different experiment, and would give the smaller model roughly
> 2.5x the steps.

10,000 steps x 16 sequences x 512 positions = 81,920,000 token positions per run. **VERIFIED**

## 5. Pilot evidence (feasibility only, not a quality result)

Measured by `scripts/pilot_6_2a.py` at commit `842cbcb`: 200 steps per architecture, the first 20
discarded as warm-up, one process per candidate, on the same 200 MB stream.

| candidate | parameters | ms/step | params vs d128 | ms/step vs d128 |
|---|---|---|---|---|
| d128 | 1,767,424 | 166 | 1.0x | 1.0x |
| d224 | 4,641,280 | 253 | 2.6x | 1.5x |
| d352 | 10,537,472 | 422 | 6.0x | 2.5x |
| d480 | 18,792,960 | 745 | 10.6x | 4.5x |
| d608 | 29,407,744 | 1,197 | 16.6x | 7.2x |

- Measured step time scaled **sublinearly** with parameter count across all five candidates: at d608
  the model is 16.6x the reference's parameter count but 7.2x its step time. **VERIFIED**
- Parameter count does **not** directly determine compute. Step cost also depends on the attention
  term, which grows with context length and head count rather than with parameter count, and on
  fixed per-step overheads that do not scale at all.
- **Memory was not the binding constraint: zero swap-outs across all five candidates.** Peak GPU
  allocation reached 3.7 GB at d608 and 2.3 GB at d352, against 16 GB of unified memory. macOS
  memory pressure stayed at level 1 (normal) for every candidate except d608, which reached level 2
  (warning). The capacity arm (d352) stayed at level 1. **VERIFIED**
- **Wall-clock time was the binding constraint.** Projected d352 runtime is 1.26 h per 10,000-step
  run and 3.77 h for three seeds, excluding sample generation, checkpoint writes and official
  evaluation. The same projection gives 0.50 h for the reference architecture, while the actual
  reference runs took 32.5–39.6 min, so projections run low; expect roughly 1.4–1.6 h per d352 run.
  **INFERRED** (projection method compared against the reference arm's recorded wall-clock times)
- Within the d352 pilot the last ten timed steps averaged 450 ms against 410 ms for the first ten.
  This machine is a fanless MacBook Air, so a multi-hour run may be slower still. **VERIFIED**
  (measurement), **INFERRED** (the thermal explanation)

## 6. Primary metric

Official validation loss and perplexity, reported for **both** the full official validation file and
the clean subset. Neither is selected over the other after seeing results; both are reported
always.

Procedure, identical to the reference arm's: `eval_official_valid.py` on each run's best checkpoint,
once for the full file and once with `--clean-subset`. Best checkpoint = lowest monitoring-split loss
within 10,000 steps. The validation stream is cut into non-overlapping 512-token windows, dropping
the trailing partial window; the clean subset additionally drops the 100 contaminated stories.
**VERIFIED** (from `scripts/run_seed_replication.sh` and `eval_official_valid.py`)

## 7. Pre-registered verdict rule

"ROBUST if the two arms' ranges of official validation loss do not overlap AND the effect is at least 3x the larger within-arm SD. INCONCLUSIVE otherwise. No p-values or confidence intervals (n=3)."

Applied separately to the full validation file and to the clean subset, in the wording above, with
no other criterion.

**Sign convention:** effect = mean(reference arm loss) - mean(capacity arm loss). A **positive**
effect means the capacity arm is better.

No additional significance criteria will be invented after seeing results.

| target | ranges overlap? | effect | larger within-arm SD | effect / larger SD | verdict |
|---|---|---|---|---|---|
| full validation file | to be filled | to be filled | to be filled | to be filled | to be filled |
| clean subset | to be filled | to be filled | to be filled | to be filled | to be filled |

## 8. Prediction

**Predicted: d352 achieves lower validation loss.** This is a prediction, not a guaranteed outcome.

**Interpretation condition:** If d352 is worse, the result will not be reported as "capacity does not
help". It will be reported as **"capacity at this LR policy does not help"**, because the wider model
may interact differently with the fixed 1e-3 learning-rate policy.

Such a result would make the 6.3 LR-policy experiment a **prerequisite for interpreting the capacity
question**, rather than evidence that capacity is ineffective.

## 9. Provenance — mandatory precondition, not an optional audit

Recorded at drafting time. The pre-training check compares against these values rather than
re-deriving them.

| item | sha256 |
|---|---|
| `train_transformer.py` at `c4db503` | `817e03ecb9bac9491c904c3ae5311540b55fc6b35dfd777f4f5b916fc3b91c76` |
| `model/transformer.py` at `c4db503` | `63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0` |
| `data/stage6_1_train_tokens.npy` (200 MB training stream) | `5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb` |
| `data/stage3_val_tokens.npy` (monitoring stream) | `9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e` |
| `tokenizer/bpe_4000.json` | `e3876f558896533d13b7818282c6b19b55cbf205390ad4ee9e982d105762d3df` |
| `eval_official_valid.py` pre-change, used for the reference arm's official evals | `06293fa4de5085432703be40e99e7f88830a9d126733b57275f740becd51c931` |

Both source files are byte-identical in the working tree at HEAD `842cbcb` to their content at
`c4db503`, so no change has entered since the reference arm was trained. **VERIFIED**

### When the section 10 code change lands

The post-change hashes of `train_transformer.py` and `model/transformer.py` are **ADDED** to this
section as the **capacity arm's** provenance. The `c4db503` hashes above are **not replaced**: they
remain the **reference arm's** provenance, because that is the code the reference runs were trained
under. The equivalence evidence linking the two sets of hashes is recorded alongside them, and the
whole change is documented under section 13 **before** any training.

### Capacity-arm provenance (added 2026-09-22, option 1 implemented)

| item | sha256 |
|---|---|
| `train_transformer.py` post-change | `57d4e57eb7d6e6bff6c8042e2842109b0cd2f72949ef0b3ee21548842fe393a5` |
| `model/transformer.py` | `63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0` — unchanged by the edit |
| `eval_official_valid.py` post-change | `ccc7bb9bd0b98aa2d2349e58329188db17d11e3a47686d77ad9c58edd187310a` |

`eval_official_valid.py` defines the primary metric, so it is frozen under the same discipline as
the training code. Its omission from the original list was an oversight in this document, not a
licence to change it freely; both its pre- and post-change hashes are recorded, and the pre-change
hash is the one the reference arm's published official losses were produced under.

The reference arm's `c4db503` hashes above stand unaltered. `model/transformer.py` is byte-identical
across both arms; only `train_transformer.py` differs, by the additive flags in section 10.

**Equivalence evidence:** `results/equivalence_6_2/`. At default arguments, seed 1337, 6.1 config,
1,001 steps on CPU, the post-change code reproduced the pre-change code bitwise on per-step loss,
per-step gradient norm, per-step batch start positions, the whole per-step log, the initial-weights
and eval-batch fingerprints, the final weights, and the step-1,000 generation text and draw count.
On MPS the deterministic quantities (initial weights, eval batches, the running batch-start hash
over all 1,001 steps) are likewise bitwise equal. **VERIFIED**

### Before any d352 training

1. Verify the repository tree is clean.
2. Verify `train_transformer.py` matches the recorded hash.
3. Verify `model/transformer.py` matches the recorded hash.
4. Record the exact git commit and those file hashes in each d352 run's provenance.
5. Verify the 200 MB corpus hash matches the recorded value.
6. Verify tokenizer identity/hash.
7. Verify configuration and optimizer settings match the registered design.
8. **If `train_transformer.py` or `model/transformer.py` has changed, STOP.** Do not train d352.
   Report that the reference arm must be rerun before the comparison can proceed.

## 10. OPEN ITEM — the capacity arm cannot be launched under the current code

**Registered resolution: option 1 below. Decided and IMPLEMENTED 2026-09-22; equivalence verified.**
Section 9 carries the capacity-arm hashes and the evidence pointer; section 13 records the
amendment. The evidence is in `results/equivalence_6_2/`.

**Deviation to note.** The bitwise comparison was run on **CPU**, not MPS. MPS is not
bit-reproducible run to run on this machine: rerunning the recorded `sr_data200mb_s1337` run's first
1,001 steps with unchanged code and the same seed reproduced every batch start position but diverged
in loss from step 5, so no code change can be verified bitwise on MPS. CPU determinism was confirmed
with an identical-code control before it was relied on. The MPS quantities that are deterministic
were compared on MPS and match. **VERIFIED**

`train_transformer.py` constructs the model as `TinyTransformer()` with no architecture arguments,
so d_model, d_ff and head count come from the constants in `model/transformer.py` (d_model 128,
d_ff 512). **There is no way to train d352 without changing one of the two files frozen in section
9**, which trips the STOP rule in step 8. **VERIFIED** (from the source at HEAD `842cbcb`)

The three options considered, with option 1 selected:

1. **Additive flags, behaviour-preserving (recommended).** Add `--d-model`, `--n-heads`, `--d-ff`
   arguments to `train_transformer.py`, defaulting to the current constants, so a default
   invocation is bit-identical to the code the reference arm ran.

   **Equivalence verification uses the 1,001-step prefix harness from the Step B/D protocol work,
   not a shorter run.** The prefix must extend past step 1,000 so that it covers the step-1,000
   in-training generation and the RNG call ordering around it — the exact place where a change to
   model construction or argument handling could shift the RNG call ordering. (Since the Step C
   change, in-training generation draws from its own generator, so this check confirms that
   decoupling still holds after the edit rather than assuming it.) Compared against the
   pre-change code at default arguments, all four must match:
   - per-step loss;
   - per-step gradient norms;
   - per-step batch start positions;
   - final weights.

   The post-change hashes are then **added** to section 9 as capacity-arm provenance, leaving the
   `c4db503` hashes in place as the reference arm's, with the equivalence evidence recorded to link
   them, and the change documented under section 13 **before** any training. The reference arm is
   then not rerun, justified by demonstrated bitwise equivalence rather than by an unchanged hash.
2. **Rerun the reference arm.** Not selected. Change the code freely, then retrain all three
   reference seeds under the new code. Costs about 1.7 h of additional compute and makes both
   arms new.
3. **Separate launcher.** Not selected. Train the capacity arm from a new file that does not touch
   the two frozen files. Rejected because it would duplicate the training loop and could
   silently diverge from the protocol, which is the risk the frozen hashes exist to prevent.

Option 1 preserves the reuse of the reference arm and keeps the guarantee that matters — identical
training semantics — while replacing hash equality with demonstrated bitwise equivalence.

## Secondary generation evaluation — PENDING CATEGORY 2 RELIABILITY

**Status: resolved by Amendment 3 (section 13), 2026-09-23, before any training.** The text as
originally registered is kept here for the record, followed by the schema it deferred.

> The secondary schema is **not** locked here, and no Category 2 analysis is invented in advance. It
> will be finalized only after Pass 4 vs Pass 5 reliability has been assessed on the sealed v2 package.
> The primary validation-loss protocol in sections 6 and 7 is already locked and does not depend on
> that outcome.

### Schema (Amendment 3)

**Category 2 survives.** On the sealed v2 package (commit `c80ad6a`), passes 4 and 5 agreed on
**301 of 320 judgments (94.1%)**, with **64 of 80** generations scored identically on all four items.
Details in `eval/category2_retest/NOTES.md`.

**Primary secondary metrics**, reported for both arms:

| item | Cohen's kappa (pass 4 vs 5) | agreement |
|---|---|---|
| `unexplained_object` | 0.87 | 93.8% |
| `uncaused_action` | 0.77 | 95.0% |
| `character_discontinuity` | 0.88 | 93.8% |

**`contradicted_ending` is reported but flagged as low-reliability** and carries no claim on its own:
kappa 0.51, flagged only 7 and 4 times out of 80, and the v2 package contains no positive worked
example of it (see the known limitation in `eval/category2_v2_examples/package_notes.md`).

**Procedure.** Each arm — reference arm (reused) and capacity arm (new) — gets **two independent
scoring passes**, each in a fresh incognito chat, with the sealed v2 package, the same opening
message (`eval/category2_retest/opening_message.txt`) and the same model. Four passes in total. For
each item and each arm, report the **mean of the two passes** and the **between-pass spread**.

**The rubric is reused; the generations are not.** The 80 items in the sealed v2 package's
`generations.json` come from the Stage 6.1 comparison and are NOT the 6.2 arms, so a new blind set
is built from the 6.2 arms' checkpoints by the same procedure as the 6.1 set:

- generated with the same settings as the 6.1 set: the 8 prompts in `eval/prompts.json`, top-k 40,
  temperature 1.0, seeds 1337-1341, a 150-token cap, 40 generations per model;
- shuffled and label-free, carrying no indication of which arm produced an item;
- with its own sealed map from blind id to source, sealed before scoring and not opened until the
  scoring passes are complete;
- hashed before scoring, with the hashes recorded and verified at the start of every pass;
- `checklist.md`, `examples.md`, `README.md` and `output_format.json` carried over **byte-identical**
  from the sealed v2 package (hashes in `eval/category2_v2_examples/package_notes.md`), so only
  `generations.json` differs between the two packages.

The kappa values were measured on Stage 6.1 text and are assumed, not demonstrated, to hold on the
new 6.2 blind set. This assumption is not separately re-tested, because the design already provides
the check: with two passes per arm, four passes over the new blind set, between-pass agreement is
measured as part of the experiment. If the observed between-pass agreement on the new blind set falls
materially below the v2 baseline (94.1% overall, per-item 93.8-95.0%), the secondary results are
reported as unreliable rather than interpreted. This judgment is made from the agreement figures
alone, before any comparison between arms.

**No verdict rule is pre-registered for the secondary metrics.** They are exploratory. Only official
validation loss carries the ROBUST / INCONCLUSIVE rule in section 7, and no secondary result can
change that verdict.

## 11. Stopping rule

**"Three seeds per arm are the complete experiment. No additional seed will be added after seeing
the results."**

## 12. Interpretation boundaries

**This experiment CAN test:** whether the larger model improves validation loss under fixed data
exposure, fixed training steps, a fixed optimizer and LR policy, and fixed tokenizer, context length
and batch size.

**This experiment CANNOT establish:**
- which model is more compute-efficient;
- which model is better at a fixed FLOP or token budget;
- whether a different LR policy would change the capacity result.

## 13. Changes after preregistration

Any modification to the locked design requires explicit documentation **before** training, recording
what changed and why. No change may be made silently after observing results.

### Amendment 1 — 2026-09-22, before any training

**What changed.** `train_transformer.py` gained three additive arguments, `--d-model`, `--n-heads`
and `--d-ff`, each defaulting to the corresponding constant in `model/transformer.py`. The model is
now constructed from those arguments, and the run config records them plus the resulting parameter
count. No RNG-consuming call was added, removed or reordered, and nothing before the first random
draw changed. `model/transformer.py` was not modified.

**Why.** Section 10: the capacity arm cannot be built otherwise, because the pre-change code
constructed the model with no architecture arguments.

**Verification.** The 1,001-step prefix harness at default arguments, seed 1337, 6.1 config, covering
the step-1,000 in-training generation. Every compared quantity was bitwise equal (section 9 and
`results/equivalence_6_2/`), and the batch-start stream was additionally reproduced independently to
confirm the generation still draws only from its own generator and leaves step 1,001 untouched.

**Consequence.** The reference arm is NOT rerun. Its provenance stays at `c4db503`, justified by
demonstrated bitwise equivalence rather than by an unchanged file hash.

**Mechanical check.** A 50-step smoke test at `--d-model 352 --n-heads 4 --d-ff 1408` built exactly
**10,537,472** parameters with d_ff 1408, 4 heads, 6 blocks, context 512, vocab 4,000, matching the
registered capacity arm in section 2. Scratch output only; this was not training.

### Amendment 2 — 2026-09-22, before any training

**What changed.** `eval_official_valid.py` now builds the model from the checkpoint's own `config`
(`d_model`, `n_heads`, `d_ff`), falling back to `model/transformer.py`'s constants when a checkpoint
carries no such keys. Nothing else changed: no scoring, windowing, subset or reporting logic, and no
output format.

**Why.** The pre-change code built `TinyTransformer()` with default dimensions and then loaded the
checkpoint into it, so it raised a shape mismatch on any d352 checkpoint. Every capacity-arm run
would have trained for about 1.7 h and then failed at evaluation.

**Verification.** Both official evaluations were rerun on the reference-arm checkpoint
`sr_data200mb_s1337_best.pt` with the post-change code:

| evaluation | committed | rerun | result |
|---|---|---|---|
| full file | 2.1065 (ppl 8.22), cross-check 2.1065 over 4,796,928 tokens | 2.1065 (ppl 8.22), cross-check 2.1065 over 4,796,928 tokens | match |
| clean subset | 2.1080 (ppl 8.23), cross-check 2.1079 over 4,766,208 tokens | 2.1080 (ppl 8.23), cross-check 2.1079 over 4,766,208 tokens | match |

Both regenerated reports are byte-identical to the committed ones apart from their timestamp line
(`results/equivalence_6_2/regression_official_*.md`). Reference-arm checkpoints carry no architecture
keys and take the fallback path, so they are scored by exactly the model they were trained with; a
d352 checkpoint loads and reports 10,537,472 parameters. **VERIFIED**

**Consequence.** The reference arm is NOT rerun, and its published official losses stand unchanged,
since the rerun reproduces them exactly.

### Amendment 3 — 2026-09-23, before any training

**What changed.** The section headed "Secondary generation evaluation — PENDING CATEGORY 2
RELIABILITY" was resolved: the secondary schema is now set out there, and the section's original
text is preserved above it.

**Why.** That section deferred the schema until Pass 4 vs Pass 5 reliability had been measured on
the sealed v2 package. It has now been measured, so the condition is discharged.

**Evidence.** Passes 4 and 5, scored in fresh incognito chats on the sealed v2 package with the same
opening message and model, agreed on 301 of 320 judgments (94.1%), with 64 of 80 generations scored
identically on all four items; per-item kappa 0.87, 0.77, 0.88 and 0.51. Files and validation:
`eval/category2_retest/pass4_scores.json`, `pass5_scores.json` and `NOTES.md`, commit `01ca3bf`.
**VERIFIED**

**Known gap, carried forward.** The v1 passes 2 and 3 were never recovered, so no v1 test-retest
figure exists and the rubric-change diagnostics (v1 vs v2) are impossible, not deferred. This does
not affect the schema above, which rests only on the v2 measurement.

**Consequence.** No change to the primary metric, the verdict rule, the arms or the training
protocol. Nothing about the capacity comparison in sections 6 and 7 depends on this section.
