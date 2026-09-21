# Stage 6.1 — design document

_Design only. Nothing has been downloaded, implemented, trained, or evaluated._

## 1. Research question

Stage 6 asks which intervention most improves entity and causal consistency under a
controlled evaluation: **more training data**, **greater model capacity**, or **a
different context length**. Stage 6.1 tests the first.

## 2. The intervention, stated precisely

**The training compute budget and optimization procedure are held constant while the
amount of unique training data available to the model is increased.**

- Same 10,000 steps, same batch 16, same 512-token context: the model sees
  81,920,000 training tokens in both runs.
- The baseline draws those tokens from ~4.69M unique tokens (≈17.5 passes). Stage 6.1
  draws them from a projected ~49M unique tokens (≈1.7 passes).

**What this does not isolate.** Sampled batches necessarily differ between the two
runs, because the pool they are drawn from differs. The intervention therefore
conflates "more unique data" with "fewer repetitions of each example" — those two are
inseparable at a fixed token budget, and that is inherent to the question being asked,
not an oversight. This is the cleanest practical isolation of data availability, not
isolation of a single microscopic variable.

## 3. Baseline

`checkpoints/transformer_10k_best.pt` — step 9,750, official validation perplexity
**9.20** (loss 2.2197). Explicitly **not** the 10,000-step checkpoint.

Baseline reference numbers, all from existing tracked results:

| Quantity | Value | Source |
|---|---|---|
| Official validation loss / PPL | 2.2197 / 9.20 | `results/stage4_transformer_10k.md` |
| ^ measured on the **full** validation file, which contains 92 stories the baseline trained on (§9.4) | — | `metadata/stage6_1_corpus.json` |
| Monitoring-split loss | 2.2031 | `results/stage4_transformer_10k.md` |
| Name retention among named generations | 13/25 (52%) | `eval/stage5_error_analysis.md` |
| Mean causal flags per generation | 3.33 of 4 | `eval/stage5_error_analysis.md` |
| At least one causal flag | 40/40 (100%) | `eval/stage5_error_analysis.md` |
| Reached `<EOS>` | 35/40 (88%) | `eval/stage5_error_analysis.md` |
| Grammatical | 27/40 (68%) | `eval/stage5_error_analysis.md` |
| Invented words | 9/40 (22%) | `eval/stage5_error_analysis.md` |
| Mean 4-gram repetition | 2.0% | `eval/stage5_error_analysis.md` |

## 4. Fixed configuration

Each item below is confirmed identical to the baseline run.

| Item | Value | How it stays fixed |
|---|---|---|
| Architecture | 1,767,424 params, T=512, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied head | `model/transformer.py` unmodified |
| Tokenizer | `tokenizer/bpe_4000.json` | Loaded frozen; **not retrained** on the new corpus |
| Optimizer | AdamW, lr 1e-3, weight decay 0.01 on projection/FFN weights only | `make_param_groups` unmodified; grouping printed at startup as before |
| Gradient clipping | 1.0, before every step | `train_step` unmodified |
| Training length | 10,000 steps, batch 16 → 81,920,000 tokens | Same CLI values |
| Seed policy | 1337; eval batches from `make_fixed_batches(seed=1337)` | Unmodified |
| Monitoring split | The existing 1,109 Stage 3 stories, excluded from training | Reconstructed and verified by hash (§7) |
| Official validation | `data/tinystories_valid.txt` untouched until the final eval | Not read by any 6.1 training or prep step |
| Stage 5 harness | `eval/prompts.json`, top-k 40, seeds 1337–1341, 150-token cap, committed rules in `eval_error_analysis.py` | Rules unchanged; only a `--checkpoint` override added (§11) |
| Checkpoint selection | Best monitoring-split checkpoint within 10,000 steps | Same policy that selected step 9,750 |

## 5. Corpus extraction

### 5.1 What we know about the existing subset

| Property | Value |
|---|---|
| Size | **19,999,022 bytes** |
| Characters | 19,985,075 |
| Difference | 13,947 bytes |
| Non-ASCII | 6,986 chars = 25 two-byte (+1 each) + 6,961 three-byte (+2 each) = **13,947** |
| Final bytes | `e end.\n<|endoftext|>` — ends exactly on the separator, no trailing newline |
| Stories | 22,174 |

The arithmetic closes exactly, so the character count is fully explained and the file
is byte-intact. **The character count (19,985,075) is not the resume offset.** If the
subset is a byte-prefix of the official file, the resume offset is its byte length,
**S = 19,999,022**. Step 5.3 verifies that premise rather than assuming it.

### 5.2 Pin the source

1. `HEAD` the official file at
   `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt`;
   record `Content-Length`, `ETag`, and `Accept-Ranges`.
2. Resolve `main` to a commit SHA and issue **all** subsequent Range requests against
   the pinned revision URL (`.../resolve/<sha>/TinyStories-train.txt`), so the byte
   offsets remain meaningful if the branch later moves.
3. Record the URL, SHA, ETag and Content-Length in the manifest (§6).

### 5.3 Determine and verify the byte offset

Two independent probes, both of which must pass:

- **Head probe.** Range `bytes=0-4095`; must equal the local subset's first 4,096
  bytes.
- **Seam probe.** Range `bytes=19994926-20003117` (S−4096 through S+4095); its first
  4,096 bytes must equal the local subset's **last** 4,096 bytes.

Both matching establishes that the local file is a prefix of the official file and
that S is the exact continuation offset. Then two boundary assertions on the seam
probe's second half:

- byte at offset S is `\n` (0x0A), since our file ends on `<|endoftext|>`;
- bytes `S+1 .. S+13` are **not** `<|endoftext|>`, i.e. the next thing is a story, not
  a second separator.

**Fallback if the head or seam probe fails.** The subset is then not a prefix at S.
Download the first ~25 MB of the official file, locate the local file's final 4,096
bytes as a byte substring, and take the offset immediately after that match as S.
Record the discovered offset, the method used, and the match count (must be exactly 1)
in the manifest. Do not proceed if the match count is not 1.

### 5.4 Fetch and trim

- Target total corpus: **200,000,000 bytes** (decimal MB, stated explicitly to avoid
  MB/MiB ambiguity). Requested range is `bytes=19999022-199999999`, i.e. ~180 MB new.
- Verify the response is HTTP `206`, that `Content-Range` matches the request, and that
  the received length equals the requested length.
- **Trim** the fetched chunk back to and including its final `<|endoftext|>`,
  discarding the trailing partial story. Record the trimmed end offset E′ and the
  number of bytes discarded.
- Assemble `data/tinystories_train_200mb.txt` = local subset bytes ++ trimmed chunk.
  Assert its size equals S + len(trimmed). Record its SHA-256.

Because the chunk begins at exactly S and the subset is a verified prefix, the
concatenation reproduces the official file's first E′ bytes byte-for-byte: there is no
gap and no overlap by construction. §5.5 verifies that at the story level as well.

### 5.5 Seam and duplication verification

Split the combined corpus on `<|endoftext|>` and normalize each story by stripping
leading/trailing newlines only — the same normalization `prepare_data.py` uses.

| Check | Expected |
|---|---|
| Total story count | 22,174 + N_new |
| Stories 1..22,174 hash-identical to the baseline subset's stories | all match |
| Story 22,175 ≠ story 22,174 | no seam duplication |
| Seam region renders as `...\n<|endoftext|>\n...` | exactly one separator, one newline each side |
| Exact-duplicate story texts across the whole corpus | **reported, not assumed zero** — TinyStories may contain genuine repeats |

### 5.6 Overlap checks

SHA-256 over normalized story text, against three reference sets:

| New stories vs | Expected | If non-zero |
|---|---|---|
| Baseline's 21,065 training stories | 0 | report count (would indicate a seam error) |
| The 1,109 monitoring-split stories | 0 | **remove those stories from 6.1 training**, count recorded in the manifest |
| The 21,990 official validation stories | 0 | **remove those stories from 6.1 training**, count recorded in the manifest |

**Resolved (decision 5).** Exact duplicates of any monitoring-split or official
validation story are *removed* from the training corpus, with counts recorded. The
first-100-normalized-character near-duplicate probe is **reported only** and does not
trigger removal, since a shared opening sentence is not evidence of a duplicated
story in a synthetic corpus with formulaic openings.

Reading the official validation file for hashing is a read-only integrity check. It
produces no training signal and no metric; the file remains unused for evaluation
until the final official eval.

### 5.7 Tokenizer audit under the frozen tokenizer

Encode the full corpus with `bpe_4000.json` and record: total BPE tokens, training-only
tokens, `<UNK>` count and rate, and any characters in the new text absent from the
tokenizer's 95-character alphabet (listed explicitly).

## 6. Metadata manifest

Written to **`metadata/stage6_1_corpus.json`** — tracked, since `.gitignore` covers
`data/` but not `metadata/`.

Fields: `source_url`, `revision_sha`, `etag`, `content_length`, `retrieved_at`;
`baseline_bytes` (S), `offset_method` (prefix-verified or fallback-search),
`head_probe_ok`, `seam_probe_ok`; `requested_range`, `returned_content_range`,
`trim_end_offset`, `bytes_discarded_in_trim`; `new_bytes`, `total_bytes`,
`baseline_stories`, `new_stories`, `total_stories`, `total_characters`;
`bpe_tokens_total`, `bpe_tokens_training`, `unk_count`, `unk_rate`,
`characters_outside_alphabet`; `duplicate_story_count`, `overlap_with_baseline_train`,
`overlap_with_monitoring`, `overlap_with_official_valid`, `stories_removed`,
`near_duplicate_probe` (reported, no removal);
`sha256_baseline_subset`, `sha256_combined_corpus`; `python_version`, `torch_version`.

**Also recorded (flag 1).** `characters_per_token` and `characters_seen_at_budget`
for **both** the baseline and 6.1 — that is, `81,920,000 x (corpus characters /
corpus tokens)`. The token budget is identical by construction, so this quantifies how
much *text* each run actually sees, making any compression drift under the frozen
tokenizer visible as a number rather than an assumption.

## 7. Data preparation

`prepare_data_stage6.py` (new; `prepare_data.py` is left untouched so the baseline
remains reproducible):

1. **Reconstruct the baseline split deterministically.** Take the 22,174 baseline
   stories in file order, apply `random.Random(1337).shuffle`, take the first 1,109 as
   the monitoring split — exactly the existing logic.
2. **Verify preservation.** Re-encode that monitoring split and assert the resulting
   array is byte-identical to the existing `data/stage3_val_tokens.npy`. This proves
   the held-out set is unchanged rather than merely intended to be. Abort on mismatch.
3. **Build the 6.1 training set**: all corpus stories, minus the 1,109 monitoring
   hashes, minus any stories excluded by §5.6. Shuffle with `random.Random(1337)` and
   concatenate with one `<EOS>` per story, matching the baseline's stream construction.
4. Write `data/stage6_1_train_tokens.npy` (gitignored). The monitoring stream is reused
   unchanged — not regenerated.

**Projected scale** (to be measured, not assumed): the baseline's 20 MB yields
4,945,550 tokens, so 200 MB projects to ~49M tokens, and 81,920,000 training tokens
would be ≈1.7 passes versus the baseline's 17.5.

## 8. Training

```
train_transformer.py --run-name transformer_6_1 --steps 10000 --batch-size 16 --lr 1e-3
```

with `--train-tokens data/stage6_1_train_tokens.npy` (the one new argument, §11).
Outputs: `logs/transformer_6_1_training.csv`, `logs/transformer_6_1_grad_norms.csv`,
`checkpoints/transformer_6_1_best.pt` + periodic, `results/stage6_1.md`.

Gradient norms are logged per step and audited for spikes above 10× median, as in the
baseline runs.

**Resolved (decision 11).** The baseline reached 10,000 steps as 5,000 + a resumed
5,000; 6.1 is a **single continuous 10,000-step run**, documented as a deviation from
the baseline's procedure. Optimizer semantics are identical (AdamW state was restored
on resume) and the difference is limited to which random windows are drawn.
**Checkpoint selection: best monitoring-split validation loss across the run**, the
same policy that selected step 9,750 for the baseline.

**Resolved (decision 13).** Qualitative samples are recorded from the fixed prompt
"Once there was a little girl" at each 1,000-step mark, matching the baseline runs.

## 9. Evaluation design (fixed now, not after results)

### 9.1 Metrics reported for both models

| Metric | Source of measurement |
|---|---|
| Official validation loss and PPL, **full file** | `eval_official_valid.py`, same windowing, same stride cross-check, same baseline rows |
| Official validation loss and PPL, **clean subset** (§9.4) | same methodology, 21,890 stories |
| Name retention among named generations | Category 1, `eval_error_analysis.py` |
| Category 1 full breakdown (recurs / changes / disappears / no name, uninitiated names) | same |
| **Mean causal flags per generation** | Category 2, blind pass (§10) |
| Each checklist item's rate (4 items) | same |
| "At least one causal flag" | same — **reported but not a success criterion** (§9.2) |
| Mean generation length, `<EOS>` rate | Category 3 |
| Invented-word rate, 4-gram repetition | Category 3 |

### 9.2 Saturation

"At least one causal flag" was **40/40 (100%)** at baseline. It cannot show
improvement and will be reported for completeness only. **Mean causal flags per
generation (3.33 of 4 at baseline)** is the causal metric with room to move, along
with the four individual item rates — of which `contradicted_ending` (60%) and
`unexplained_object` (78%) have the most headroom, while `uncaused_action` (100%) is
also saturated.

### 9.3 No pre-registered thresholds

No success threshold is defined. What gets reported is fixed now; magnitude is
interpreted honestly afterwards. **With 40 generations — 25 of them named at baseline
— small shifts are within noise.** Concretely: one generation is 2.5 percentage points
of the category-1 or category-3 rates and 4 points of name retention, so a change of
"a few points" may be one or two generations changing.

### 9.4 Contamination and the clean validation subset (resolved, fixed before training)

The corpus build discovered pre-existing contamination in the baseline's data:
**92 of the baseline's 21,065 training stories are exact duplicates of official
validation stories**, and 8 monitoring-split stories are as well. The baseline's own
train/monitoring split is clean (0 overlap). This was not introduced by Stage 6.1.

Per decision 5 those stories are removed from the 6.1 training corpus, so **6.1's
training data is not a strict superset of the baseline's** — it is the baseline's data
minus 92 contaminated stories, plus the new material. The asymmetry favours the
baseline, which memorised 92 validation stories that 6.1 never sees.

**Resolved: option B.** No retraining; instead, a second evaluation target is defined
now and will not be changed afterwards.

> **Clean validation subset** = `data/tinystories_valid.txt` minus every story whose
> exact text duplicates a story in either model's training data or in the monitoring
> split.

Measured from the validation side: **100 excluded stories** (92 duplicating a baseline
training story + 8 duplicating a monitoring story; 100 distinct hashes, no
double-counting), leaving **21,890 of 21,990 stories**. The 6.1 training corpus has
zero exact overlap with the validation file by construction, so the union reduces to
those two sources. The 100 excluded hashes are recorded in
`metadata/stage6_1_corpus.json` under `clean_validation_subset.excluded_story_hashes`,
fixed before training begins.

**At final evaluation, report four numbers**: official PPL on the **full** validation
file and on the **clean subset**, for **both** the baseline (step 9,750) and 6.1 — all
four using the `eval_official_valid.py` methodology unchanged (non-overlapping
512-token windows, stride cross-check, same baseline rows). The clean-subset pass
rebuilds the story list, drops the 100 hashes, then re-windows; window boundaries
therefore differ slightly from the full-file pass, and the two are compared
like-for-like only within a column.

### 9.5 Notes to carry into the results

These three facts are recorded now so they are reported rather than rediscovered:

1. **Duplicates inside the training corpus were kept.** The corpus contains 2,950
   exact-duplicate story texts. They are left in place, matching the baseline's
   condition — the baseline's own 20 MB slice contains such duplicates too, and
   removing them from 6.1 only would introduce a second asymmetry.
2. **Near-duplicate openings are common and were not removed.** 11,377 new stories
   share a first-100-character opening with an official validation story, and 3,366
   with a monitoring story. TinyStories openings are formulaic, so a shared opening is
   weak evidence of a duplicated story; this is a residual, unquantified leakage risk
   that is reported rather than eliminated.
3. **The Stage 4/5 official perplexity of 9.20 was measured on the full validation
   file, which contains 92 stories the baseline trained on.** Every prior official-PPL
   figure in `results/stage4_transformer.md` and `results/stage4_transformer_10k.md`
   (10.21 at 5k, 9.20 at 9.75k) carries that caveat. The clean-subset number produced
   at final evaluation is the contamination-free comparison point.

### 9.6 Length check

Mean generation length is reported beside every entity and causal metric. Shorter
generations have mechanically fewer opportunities to drift or contradict, so any
apparent improvement must be checked against a fall in length before being read as a
consistency improvement.

## 10. Blind rescoring of category 2 — run in a separate session

**Resolved (flag 3).** Blind scoring is **not** performed in the session that builds
the package. This session prepares a self-contained scoring package; the user runs it
in a fresh Claude Code session with no history of this project.

1. Generate 40 new generations with the frozen harness (8 prompts × seeds 1337–1341,
   top-k 40, cap 150) from `checkpoints/transformer_6_1_best.pt`.
2. Combine with the baseline's existing 40 from `eval/generations_topk40.json` → 80.
3. **Write the sealed mapping first**: `eval/blind_map_stage6_1.json` maps
   `blind_id → {source_model, original_id, prompt_id, seed}`. Written and committed
   **before** any score exists; its SHA-256 is recorded in the run's results file.
4. **Write the blind set**: `eval/blind_set_stage6_1.json` contains only
   `blind_id`, prompt text and generation text, shuffled with `random.Random(1337)`.
   No checkpoint or source label, no ordering signal.
5. **Assemble the scoring package** `eval/blind_scoring_package/`, self-contained and
   readable without any project context:
   - `README.md` — the task, and nothing about which models exist or what is being
     compared;
   - `checklist.md` — the four category-2 rules copied verbatim from the committed
     docstring in `eval_error_analysis.py`, plus the scoring conventions already used
     (animals/people are characters, inanimate things are objects, outdoor scenery is
     generic, a generation cut off by the length cap has no ending so
     `contradicted_ending` is false);
   - `generations.json` — the 80 shuffled unlabeled items;
   - `output_format.json` — a template showing the exact required output: per
     `blind_id`, the four booleans plus an `evidence` object naming the triggering
     phrase for each true flag.
6. The fresh session returns `eval/blind_causal_scores_stage6_1.json` keyed by
   `blind_id`.
7. **Unblind** via the sealed map; compute per-model means.
8. **Report drift**: the fresh session's scores for the baseline 40 against their
   original Stage 5.5 scores, per checklist item and in mean flags per generation.
   **6.1 is compared against the fresh-session baseline scores, never against the
   original Stage 5.5 numbers.**

**What this does and does not establish.** Because the fresh session has no history of
this project, it has not seen the baseline texts before, and the drift figure in step 8
is **cross-session scorer consistency** — two independent applications of the same
checklist by the same model class, without shared context. It is not human validation
and not inter-rater agreement between different scorers. Categories 1 and 3 are
deterministic code and are unaffected by blinding.

## 11. Files created or modified

**New**

| File | Purpose | Tracked |
|---|---|---|
| `fetch_corpus_stage6.py` | HEAD/pin, probes, Range fetch, trim, assemble, verify, write manifest | yes |
| `metadata/stage6_1_corpus.json` | The manifest of §6 | yes |
| `prepare_data_stage6.py` | Split reconstruction + verification + stream build | yes |
| `blind_rescore.py` | Build blind set + sealed map; merge scores; unblind; drift report | yes |
| `eval/blind_map_stage6_1.json` | Sealed mapping, written before scoring | yes |
| `eval/blind_set_stage6_1.json` | Label-free 80-generation set | yes |
| `eval/blind_causal_scores_stage6_1.json` | Blind category-2 scores | yes |
| `results/stage6_1.md` | Training run summary | yes |
| `eval/stage6_1_comparison.md` | Final baseline vs 6.1 comparison | yes |
| `data/tinystories_train_200mb.txt`, `data/stage6_1_train_tokens.npy` | Corpus and token stream | no (gitignored) |

**Modified**

| File | Change |
|---|---|
| `train_transformer.py` | Add `--train-tokens` path override. No change to procedure, optimizer, clipping or logging. |
| `eval_error_analysis.py` | Add `--checkpoint` and output-path overrides. **Scoring rules unchanged.** |

**Unmodified:** `model/transformer.py`, `tokenizer/*`, `prepare_data.py`,
`eval_official_valid.py`, `eval_generation.py`, `eval_sampling.py`, `eval/prompts.json`.

## 12. Risks and open decisions

1. **The prefix assumption may fail.** If the subset was not cut from byte 0, S is
   wrong. Mitigated by two probes and a fallback search (§5.3); the run stops if
   neither resolves.
2. **Source drift.** If the HF file is revised, offsets shift. Mitigated by pinning the
   commit SHA and recording ETag + Content-Length.
3. **Frozen tokenizer on 10× data.** The tokenizer was trained on the first 20 MB.
   Expect a small but non-zero `<UNK>` rate on unseen characters, and slightly worse
   compression (more tokens per story). **This is a real confound:** the token budget is
   fixed at 81.92M, so worse compression means marginally less *text* per step. §5.7
   measures both so the size of the effect is known rather than assumed.
4. **Epochs fall from 17.5 to ~1.7.** Inherent to the intervention (§2), not separable
   at fixed compute. Must be stated in the results.
5. **Near-duplicate leakage.** Exact hashing catches only exact duplicates.
   **Resolved:** exact duplicates of monitoring or official-validation stories are
   removed; the first-100-character probe is reported but does not trigger removal, so
   a residual near-duplicate risk remains and is stated rather than eliminated.
6. **Monitoring split provenance.** The 1,109 monitoring stories come from the first
   20 MB only, so they are not a random sample of the enlarged corpus. Keeping them
   fixed is correct for comparability, but the monitoring loss for 6.1 measures
   held-out loss on the baseline's region of the corpus. The official validation file
   is the unbiased comparison; monitoring loss is for training curves.
7. **Blinding.** **Resolved:** scoring moves to a fresh session with no project
   history (§10), which removes the familiarity problem within this session. The
   remaining limit is that both scorers are the same model class applying the same
   checklist, so agreement measures cross-session consistency, not human validation.
8. **Sample size**: 40 generations, 25 named at baseline (§9.3).
9. **Single seed** for training and for the five generation seeds. A difference of the
   size we can detect here could plausibly be seed variance; no seed sweep is planned.
10. **Checkpoint selection policy** must match the baseline: best monitoring-split
    checkpoint within 10,000 steps, not the final step.
11. **Run structure.** **Resolved:** single continuous 10,000-step run, documented as
    a deviation; checkpoint chosen by best monitoring-split loss across the run (§8).
12. **Disk and time**: ~180 MB additional download plus a ~100 MB token stream; training
    time should be unchanged (~30–35 min at the baseline's measured rate), since the
    step count and batch size are identical.
13. **Qualitative samples.** **Resolved:** recorded at each 1,000-step mark, as in the
    baseline runs (§8).

## 13. Execution order and stop points

1. Fetch, verify, assemble the corpus; write the manifest. **Stop — review the
   manifest** (`<UNK>` rate, overlap counts, seam checks) before spending training time.
2. Prepare the token streams; verify the monitoring split is byte-identical.
3. Train 10,000 steps.
4. Official validation of the 6.1 best checkpoint **and** a re-evaluation of the
   baseline, each on both the full file and the clean subset — four numbers (§9.4).
5. Generate the 40 new generations; build the sealed map, the blind set and the
   scoring package. **Stop — the sealed map is committed before scoring, and the
   package is handed to a fresh session.**
6. Receive the fresh session's scores; unblind; report cross-session drift and the
   comparison.

Awaiting approval before step 1.
