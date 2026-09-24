# Track A conclusion — controlled experiments on a small from-scratch language model

**Status:** draft for review, not committed
**Date:** 2026-09-23
**Scope:** Stage 6.1 (data), the seed replication study, Stage 6.2 (capacity), the context-length question, and the Category 2 measurement work

---

## 1. What this study is

This project began as an attempt to build a language model from scratch in order to understand every stage of the process: tokenizer, model, training loop, evaluation. It became a small controlled empirical study of two interventions on a fixed training budget, carried out on a single laptop.

The primary contribution is **methodological**. The effect sizes reported here are not surprising in light of existing work on language-model scaling, and this study does not propose or test a scaling law. What it provides is a worked example of running such comparisons carefully at very small scale: replicated runs, verdict rules fixed before the data they judge, measurement instruments tested before they were trusted, full provenance for every run, and explicit handling of defects and limitations.

Every result below is specific to this model, this data, this training recipe and this budget. None of it should be read as a general claim about language models.

---

## 2. Setup common to all experiments

**Model.** A decoder-only Transformer written from scratch in PyTorch: 6 pre-LayerNorm blocks, learned positional embeddings, input and output embeddings tied, causal self-attention. The reference configuration is d_model 128, 4 heads, d_ff 512, for 1,767,424 parameters.

**Tokenizer.** A byte-pair-encoding tokenizer with a 4,000-token vocabulary, trained on TinyStories and frozen for the entire study. All losses are in nats per token under this tokenizer, and are not comparable with results under any other tokenizer.

**Data.** TinyStories. Stories are concatenated into one token stream with an end-of-story token between them, and training draws random 512-token windows from that stream, with replacement. Windows can therefore cross story boundaries.

**Training recipe.** AdamW, weight decay 0.01 on linear weights only, gradient clipping at 1.0, **constant learning rate 1e-3 with no warmup and no decay**, batch 16, context 512, **10,000 steps**. Every run therefore processes exactly 81,920,000 token positions.

**Checkpoint selection.** The checkpoint with the lowest loss on a fixed monitoring split (50 fixed batches, drawn with a fixed evaluation seed of 1337 that is independent of the training seed).

**Primary metric.** Loss on the official TinyStories validation file, scored once per token in non-overlapping 512-token windows. Two targets are always reported together and neither is chosen after the fact: the **full file** (4,787,559 tokens scored) and a **clean subset** that excludes 100 stories found to overlap with training data (4,756,899 tokens scored).

**Hardware.** A fanless MacBook Air (Apple M4, 16 GB unified memory), training on the GPU through PyTorch's MPS backend.

---

## 3. Headline results

| Experiment | Comparison | Held fixed | Effect (full file) | Effect (clean) | Verdict |
|---|---|---|---|---|---|
| 6.1 | 20 MB vs 200 MB training data | model, recipe, **token exposure** | +0.1182 nats, −11.15% perplexity | +0.1192 nats, −11.24% | ROBUST |
| 6.2 | 1.77M vs 10.54M parameters | data, recipe, **token exposure** (not compute) | +0.1821 nats, about −16.65% perplexity | +0.1823 nats | ROBUST |

Effects are reported as mean(baseline) − mean(intervention) for 6.1 and mean(reference) − mean(capacity) for 6.2, so a **positive** value favours the intervention/capacity arm; perplexity changes carry the sign of the underlying loss change and are computed from the mean losses, not from rounded perplexities. Both verdicts use the same rule: **ROBUST** if the two arms' ranges of validation loss do not overlap and the effect is at least 3× the larger within-arm standard deviation; **INCONCLUSIVE** otherwise. No p-values or confidence intervals are reported, because three runs per arm cannot support them.

The precise meaning of each result is set out below. Neither row should be summarised as "more data helps" or "bigger models help" without the qualifications in sections 4 and 5.

---

## 4. Stage 6.1 — fresh data versus repeated data at a fixed token budget

### What was compared

Both arms trained the same 1.77M-parameter model on exactly 81,920,000 token positions. They differed only in the size of the pool those tokens were drawn from:

| Arm | Training stream | Passes over its data |
|---|---|---|
| 20 MB | 4,691,267 tokens | about 17.5 |
| 200 MB | 49,124,477 tokens | about 1.7 |

The comparison is therefore **fresh data versus heavily repeated data under an identical token budget**. It is not a test of what happens when a model is given more training.

### Result

| Arm | Seed 1337 | Seed 1338 | Seed 1339 | Mean | Sample SD | Range |
|---|---|---|---|---|---|---|
| 20 MB | 2.2183 | 2.2447 | 2.2114 | 2.2248 | 0.0176 | 2.2114–2.2447 |
| 200 MB | 2.1065 | 2.1116 | 2.1016 | 2.1066 | 0.0050 | 2.1016–2.1116 |

(Full-file validation loss in nats. Clean-subset values differ by at most 0.0025 and give the same verdict.)

The effect is +0.1182 nats, 6.7× the larger within-arm SD, with no overlap between ranges: **ROBUST** on both targets.

The final gap between validation and training loss supports the interpretation directly. The 20 MB runs ended with a gap of about +0.25 nats in every seed; the 200 MB runs ended between +0.0015 and +0.0112. The model trained on repeated data was overfitting; the model trained on fresh data was not.

### History of this result

The original Stage 6.1 comparison used a single run per arm and reported −10.4% perplexity. It was replicated with three seeds per arm after the question was raised of whether a single-run difference could be distinguished from ordinary run-to-run variation. The replicated effect points the same way and is slightly larger (−11.15%).

Two protocol changes were made before the replication (section 7), and the historical seed-1337 runs were kept as diagnostics rather than folded into the three-run sets. Both protocol changes moved the seed-1337 result by less than one within-arm SD.

### Pre-registration status

The 3× threshold was fixed after the losses of the first three of the six replication runs had been reported, and before the remaining three — this is the user's recollection, corroborated by the committed record that the rule reached the project when 5 of the 6 runs were committed (`eval/seed_replication_results.md`). The verdict would be unchanged for any threshold up to 6.7×, so the timing could not have altered the outcome, but the rule was not fixed in advance of all the data it judged.

---

## 5. Stage 6.2 — capacity at fixed data exposure, not at fixed compute

### What was compared

| Arm | d_model | Heads | d_ff | Blocks | Parameters | Status |
|---|---|---|---|---|---|---|
| Reference | 128 | 4 | 512 | 6 | 1,767,424 | reused from the replication study |
| Capacity | 352 | 4 | 1,408 | 6 | 10,537,472 | newly trained for 6.2 |

Both arms used the 200 MB data and saw the same 81,920,000 token positions. **Training compute was deliberately not held constant.** In the pre-training feasibility pilot the capacity architecture took about 2.5× as long per step (422 ms against 166 ms). The real runs took 93.9, 97.1 and 98.7 minutes for the capacity arm, and 39.3, 39.6 and 116.5 minutes for the reference arm (200 MB data); no cause for the 116.5-minute outlier is recorded.

This is therefore a **capacity-at-fixed-data-exposure** comparison. It does not say which model is more compute-efficient, and it does not say which would win if both were given the same compute; a fixed-compute comparison would give the smaller model about 2.5× as many steps and is a different experiment.

### Result

| Arm | Seed 1337 | Seed 1338 | Seed 1339 | Mean | Sample SD | Range |
|---|---|---|---|---|---|---|
| Reference (reused) | 2.1065 | 2.1116 | 2.1016 | 2.1066 | 0.0050 | 2.1016–2.1116 |
| Capacity (new) | 1.9636 | 1.8904 | 1.9194 | 1.9245 | 0.0369 | 1.8904–1.9636 |

(Full-file validation loss in nats. Clean-subset means are 2.1081 and 1.9258.)

The effect, defined as mean(reference) − mean(capacity) so that a positive value favours the capacity arm, is **+0.1821 nats** on the full file and +0.1823 on the clean subset. That is 4.9× the larger within-arm SD, with no range overlap: **ROBUST** on both targets. The pre-registered prediction was that the capacity arm would achieve lower loss; it did.

The improvement is not confined to long context. The larger model's loss is lower at every position band in the window (section 6).

### Pre-registration status

The 6.2 design, verdict rule, prediction, stopping rule and interpretation boundaries were committed before any capacity-arm run started, and every change afterwards was recorded as a dated amendment before training.

The design was **not fully blind**. The reference arm consists of three runs trained earlier for the replication study, and their losses — including their SD of 0.0050 — were already published when the verdict rule was written. Only the capacity arm's results were unknown. Because the rule uses the larger of the two SDs, and the larger turned out to be the capacity arm's, the known reference SD did not in the end set the threshold. The asymmetry is recorded in the pre-registration and is repeated here so it is not mistaken for a fully blind design.

---

## 6. The context-length question, closed from existing data

A planned experiment on longer context windows was not run. Every training run already reports loss by position within the 512-token window, and in every run, in both 6.2 arms, loss stops improving after roughly 128 to 256 tokens:

| Positions in window | Reference arm mean loss | Capacity arm mean loss |
|---|---|---|
| 1–16 | 2.8461 | 2.7138 |
| 65–128 | 2.0897 | 1.9092 |
| 129–256 | 2.0607 | 1.8774 |
| 257–511 | 2.0629 | 1.8769 |

The last two bands differ by less than 0.01 nats in either arm, against roughly 0.8 nats gained over the first 128 positions. Because training windows are drawn from a packed stream and can span story boundaries, extending the window mostly adds text from earlier, unrelated stories. Under this model, data and packing scheme, a longer context would be expected to add little, and the question was closed without spending compute on it.

This finding is specific to TinyStories with packed windows. It says nothing about context use on longer documents.

---

## 7. Run-to-run variation, and what three runs can and cannot show

### What varies between runs

In this study a training seed controls two things: model initialization and the sampling of training windows. It does not control the monitoring batches used for checkpoint selection, which use a separate fixed evaluation seed, or the random sampling used for in-training sample generation, which has its own generator. Both separations were introduced deliberately (section 8), so that the seed affects training and nothing else.

The seed is not the only source of variation. A determinism test found that **two runs with identical code and seed on MPS diverge from step 2** at the level of floating-point rounding, and the difference compounds over training. Batches stayed identical; the arithmetic did not. The spread between runs reported here is therefore **run-to-run variation from the seed and MPS nondeterminism combined**, and the two cannot be separated with these data. It is not "seed variance".

### The spreads observed

| Configuration | Sample SD of validation loss |
|---|---|
| 20 MB data, 1.77M parameters | 0.0176 |
| 200 MB data, 1.77M parameters | 0.0050 |
| 200 MB data, 10.54M parameters | 0.0369 |

The larger model varied about 7.4× as much as the smaller one on the same data, with its three runs spanning 0.0732 nats — about two fifths (40.2%) of the entire 6.2 effect.

### Limits of three runs

A standard deviation estimated from three values is very imprecise; the true value could plausibly be several times smaller or larger. These SDs are therefore reported as **observations about these runs, not as estimates of a property of the configurations**. In particular, the difference between the spreads above is suggestive but not established. The verdicts are robust to this because the effects clear their thresholds by a wide margin, but any future experiment at the 10.54M scale should plan for at least the larger spread and would benefit from more runs.

Under the 3× rule, detecting an effect at the 10.54M scale would require roughly 0.11 nats, which is close to the size of the whole 6.1 effect.

---

## 8. Methodology

### Pre-registered verdict rules

Both headline comparisons are judged by a rule stated before the data it judges: fully before in 6.2, partially before in the replication study (section 4). The rules specify both validation targets, forbid choosing between them afterwards, fix the sign convention, and say what a surprising result would and would not mean.

### Replication

Every headline number is a comparison of three runs per arm, not of two single runs. The replication was motivated directly by the question of whether the original single-run 6.1 result could be told apart from run-to-run variation.

### Provenance and reproducibility controls

- Every run records a manifest with the git commit it launched from, a clean-tree flag, the seed and evaluation seed, all arguments, and SHA256 hashes of the data streams.
- Runs are launched by drivers that refuse to start from an uncommitted tree, refuse to overwrite existing outputs, check the manifest afterwards, and commit each run's results before starting the next. The Stage 6.2 driver additionally checks the recorded training-code, eval-code, data and tokenizer hashes before training starts, and its manifest check includes the exact parameter count; the seed-replication driver, which produced the 6.1 runs and the 6.2 reference arm, has neither of those two checks.
- All three data streams were regenerated from committed code and matched their recorded hashes byte for byte.
- Analysis documents are generated by committed scripts, and each script has a `--verify` mode that regenerates its document and confirms it is byte-identical.

### Testing the protocol, not just the results

Several problems in the original protocol were found by testing it rather than trusting it:

- **The seed also chose the monitoring batches**, so different seeds selected checkpoints on different data. Fixed by a separate evaluation seed.
- **In-training sample generation shared the batch random-number stream**, so the data order depended on where a sample happened to end. Fixed by giving generation its own generator.
- **The original 20 MB baseline had been trained as two separate processes** (5,000 steps, then a resumed 5,000), which a single-pass run cannot reproduce. It was retrained as a single pass, and the historical run kept as a diagnostic.
- **MPS cannot reproduce a run bitwise.** Because of this, the code change that added architecture flags for 6.2 was verified bitwise on CPU, where training is deterministic, over 1,001 steps covering sample generation, with the MPS-deterministic quantities (initial weights, evaluation batches, the full batch sequence) checked separately on MPS.
- **The evaluation script could not load a larger model's checkpoint.** It was changed to read the architecture from the checkpoint, and the change was regression-tested by reproducing previously published losses exactly.

### Testing and repairing a measurement instrument

The secondary metric, "Category 2", has a language model score each generated story on four yes/no consistency items.

In Stage 6.1 it reported 2.33 versus 2.08 flags per story, but the same stories scored in a different session differed by 1.00 flags per story — four times the difference between the models. Investigation showed the two sessions were not comparable: the second session was non-blind and its instructions were never saved, though its conventions were recorded in `eval/causal_scores.json`. So the 1.00 is a confounded observation rather than a measured noise level.

For `character_discontinuity` specifically, the counts that exist in committed files are: 46/80 under the Stage 6.1 rubric (the sealed pass in `eval/blind_scores_stage6_1.json`), and 40/80 and 35/80 under the rewritten v2 rubric (the validated pass 4/pass 5 pair). The planned retest of the original rubric (passes 2 and 3) was never recovered in usable form, so no v1 test-retest number, and no evidence-quote comparison for that item, exists.

The rubric was rewritten with a narrower, checkable definition of that item, eight worked examples chosen blind from training samples, and a sealed, hashed package. Two independent passes over the same 80 stories under the new rubric agreed on 301 of 320 judgments (94.1%), with Cohen's kappa of 0.87, 0.77 and 0.88 on three items and 0.51 on the fourth. That fourth item, `contradicted_ending`, fires rarely (7 and 4 times in 80) and has no positive worked example, and is treated as low-reliability.

The first-round retest (passes 2 and 3 under the old rubric) was never saved in usable form, so the old rubric's own test-retest agreement was not measured and no direct before-and-after comparison is reported.

### Handling of defects

Defects were recorded rather than silently corrected. The clearest example is the architecture header in section 10.

---

## 9. What this study does not establish

**Better stories.** Both headline results are reductions in next-token prediction loss. Lower loss does not by itself mean more coherent or more consistent stories. In Stage 6.1 the story-consistency metric gave only weak evidence of improvement, and for Stage 6.2 it has not yet been measured. This study establishes better next-token prediction under these conditions, not better story quality.

**Compute efficiency.** Stage 6.2 did not hold compute constant. Nothing here says whether a larger model is a better use of a fixed amount of compute.

**Behaviour at convergence.** Neither model was trained to convergence. All three capacity runs, and two of three reference runs, selected the final step (10,000) as their best checkpoint, and validation loss was still falling. Every result applies to a **10,000-step budget**; the size, and even the ranking, of the effects could differ with longer training.

**Generality.** One small architecture family, one dataset, one tokenizer, one constant learning rate, one machine. The effects are consistent with what larger studies of language models report, but this study adds no evidence that they generalise, and it neither proposes nor tests a scaling law.

**The learning-rate interaction.** A constant learning rate of 1e-3 with no warmup was used throughout, and larger models are often more sensitive to learning rate than smaller ones. Whether a different learning-rate policy would change the capacity result is unknown.

---

## 10. Known defect: the architecture header in per-run result files

There are two separate things to distinguish here.

**1. The analysis-document note (done).** Every capacity-arm run's summary file states that the model had d_model 128 and 1,767,424 parameters, which is false. The Stage 6.2 results document now carries a note explaining this. The note is produced by the analysis script rather than written into the output by hand, so that `--verify` still reproduces the document byte for byte.

**2. The hardcoded line in `train_transformer.py` (not yet fixed).** The cause is a line in the training script that writes the reference architecture into every run summary regardless of the architecture flags. The capacity runs were genuinely d_model 352 with 10,537,472 parameters: their manifests say so, every checkpoint config records the exact parameter count, and the driver refused to commit any run that did not match. No number in any analysis is taken from the incorrect line.

Fixing it means editing `train_transformer.py`, a file whose hash is frozen in section 9 of the 6.2 pre-registration and against which the capacity arm was trained. It is therefore recorded as a known defect there, and will be fixed by a documented amendment **before any Stage 6.3 training**, generating the line from the run's own arguments. That amendment will record a new hash for the training script alongside the existing ones rather than replacing them. The already-committed run summaries are left unchanged as part of the historical record.

---

## 11. Possible next steps

None of these is required for the conclusions above.

**Secondary Category 2 scoring for 6.2.** Build a new blind set of generations from both arms under the same settings as before, and score it with two passes per arm using the validated rubric. This would test whether the capacity arm's lower loss corresponds to more consistent stories.

**Stage 6.3, learning-rate policy.** Compare the constant learning rate with warmup followed by cosine decay at the same peak rate. The main design decision is the model size. At 1.77M parameters an effect of about 0.015 nats would be detectable, at a cost of about 2.0 hours per arm (3 runs); at 10.54M only an effect of about 0.11 nats would be detectable, at a cost of about 4.8 hours per arm and about 9.7 hours total (6 runs), but the result would apply directly to the larger model. The architecture-header fix must be made first.

**More runs per arm**, to estimate run-to-run spreads with useful precision, particularly at the 10.54M scale.

**A fixed-compute comparison**, and **longer training**, to address the two largest gaps in section 9.

---

## 12. Summary

Under a fixed budget of 81.9 million training tokens and 10,000 steps, on TinyStories with a from-scratch model on a single laptop:

- fresh data produced lower validation loss than heavily repeated data (+0.118 nats, −11.2% perplexity), and the repeated-data model showed clear overfitting;
- a model with about six times the parameters produced lower validation loss than the smaller one on the same data exposure (+0.182 nats, about −16.65% perplexity), at roughly 2.5× the compute per step;
- both effects held across three runs per arm and passed rules stated before, or largely before, the data they judged;
- neither model used much context beyond about 128 to 256 tokens;
- the larger model varied noticeably more from run to run, though three runs cannot pin that down.

These are statements about next-token prediction under one specific setup. The more durable output of the study is the process that produced them: replicated comparisons judged by fixed rules, a protocol that was tested and repaired where it failed, a measurement instrument that was found unreliable and rebuilt before being used again, and a record that states its own defects and limits.
