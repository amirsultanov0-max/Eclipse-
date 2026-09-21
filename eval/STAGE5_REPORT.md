# Stage 5 — benchmark report

_Synthesis of existing results. No new generation, scoring, or experiments were
produced for this document._

**Sources.** Every number below traces to one of:

| Source | Contributes |
|---|---|
| `results/stage4_transformer_10k.md` | perplexity, official validation, position analysis |
| `eval/stage5_generation_eval.md` | greedy decoding baseline, 8 prompts x 2 checkpoints |
| `eval/stage5_sampling_eval.md` | 130 generations across greedy + 4 sampling strategies |
| `eval/stage5_error_analysis.md` | structured error analysis, 40 generations |

The model under evaluation is the 1.77M-parameter Transformer (512-token context,
d_model 128, 4 heads, 6 Pre-LN blocks, tied LM head), at two checkpoints: **5k**
(`transformer_best.pt`, step 5,000) and **9.75k** (`transformer_10k_best.pt`,
step 9,750).

---

## 1. Summary

On the reserved official validation file, the 9.75k checkpoint reaches **2.2197 nats,
perplexity 9.20**, against 105.02 for the Stage 3 trained bigram and 43.32 for the
count-based bigram floor. Under greedy decoding the model repeats heavily on some
prompts (4-gram repetition 29.7% mean at 9.75k, up to 82.3% on a single prompt),
and every sampling strategy tested reduced that repetition to 1.5–5.7% mean,
indicating that the looping is strongly decoding-dependent. The structured error
analysis of 40 generations found surface-form measures substantially stronger than
entity and causal measures: 88% reached `<EOS>`, 68% were scored grammatical and
mean repetition was 2.0%, while 100% triggered at least one causal-consistency flag
and 52% of generations containing a name retained the initial name. Within this
evaluation, local language and surface form are the model's comparatively stronger
capability, and cross-sentence entity and causal state its comparatively weaker one.
The limitations in section 5 bound how far these observations generalize.

---

## 2. Perplexity

_Source: `results/stage4_transformer_10k.md`._

Full deterministic pass over `data/tinystories_valid.txt`, reserved and unused until
Stage 4: 21,990 stories, non-overlapping 512-token windows, **4,787,559 tokens scored
exactly once** (99.80% of the file). Every row below is measured on this same file
over the same token pairs.

| Model | Loss (nats) | Perplexity |
|---|---|---|
| Uniform, log(4000) | 8.2940 | 4000.00 |
| Unigram (no context) | 6.0360 | 418.24 |
| Stage 3 trained bigram (1-token context) | 4.6542 | 105.02 |
| Count-based bigram floor (add-0.01) | 3.7686 | 43.32 |
| Transformer, 5k (512-token context) | 2.3236 | 10.21 |
| **Transformer, 9.75k (512-token context)** | **2.2197** | **9.20** |

The bigram rows carry 1 token of context and the Transformer rows up to 512, so this
is an architectural comparison rather than a like-for-like one.

**5k vs 9.75k.** The additional 5,000 training steps moved official validation loss
by **−0.1039 nats**, a 9.9% perplexity reduction (10.21 → 9.20). On the Stage 3
monitoring split the same comparison is 2.3079 → 2.2031, a −0.1048 change, so both
splits measured the improvement to within 0.001 nats of each other.

**Per-1,000-step improvement**, monitoring split, showing the rate of change:

| Segment | Val loss | Improvement |
|---|---|---|
| 4k → 5k | 2.3079 | −0.0535 |
| 5k → 6k | 2.2693 | −0.0386 |
| 6k → 7k | 2.2481 | −0.0212 |
| 7k → 8k | 2.2294 | −0.0186 |
| 8k → 9k | 2.2210 | −0.0084 |
| 9k → 10k | 2.2074 | −0.0136 |

**Loss by position in the window** (9.75k, official validation):

| Target positions | Loss | Perplexity |
|---|---|---|
| 1–16 | 2.9134 | 18.42 |
| 17–64 | 2.3311 | 10.29 |
| 65–128 | 2.1998 | 9.02 |
| 129–256 | 2.1785 | 8.83 |
| 257–511 | 2.1810 | 8.85 |

Measured improvement flattens after roughly 128–256 tokens of available context, and
positions 257–511 score marginally worse than 129–256.

---

## 3. Generation

### 3a. Greedy baseline

_Source: `eval/stage5_generation_eval.md`. 8 fixed prompts, greedy (argmax) decoding,
cap 150 tokens, 1 deterministic generation per prompt per checkpoint._

| Metric | 5k | 9.75k |
|---|---|---|
| 4-gram repetition rate | 46.9% | 29.7% |
| Generation length (tokens) | 111.5 | 82.4 |
| Stopped at `<EOS>` | 5/8 | 5/8 |
| Name tracking | recurs 2, disappears 1, no name 5 | recurs 2, disappears 1, no name 5 |
| Quotation marks balanced | 6/8 | 7/8 |
| Ends on terminal punctuation | 5/8 | 5/8 |

Per-prompt repetition under greedy decoding was highly uneven: 0.0% on `girl_store`
and `rainy_friends` for both checkpoints, against 85.0% (5k) and 79.6% (9.75k) on
`tom_ball`, and 70.1% (5k) and 82.3% (9.75k) on `lily_key`. Across the 8 prompts, 5
improved at 9.75k, 1 worsened, and 2 tied.

### 3b. Sampling comparison

_Source: `eval/stage5_sampling_eval.md`. 130 generations. Seeds
`torch.manual_seed(1337 + sample_index)`, so samples 1–3 use 1337/1338/1339
identically across every strategy and prompt._

9.75k checkpoint, mean 4-gram repetition across all 8 prompts:

| | greedy | temp 0.7 | temp 1.0 | top-k 40 | top-p 0.9 |
|---|---|---|---|---|---|
| Mean repetition | 29.7% | 5.7% | 1.6% | 1.5% | 2.7% |
| Mean length | 82 | 94 | 111 | 108 | 102 |
| Stopped at `<EOS>` | 5/8 | 19/24 | 18/24 | 19/24 | 18/24 |

On the two prompts that looped worst under greedy decoding:

| Prompt | Ckpt | greedy | temp 0.7 | temp 1.0 | top-k 40 | top-p 0.9 |
|---|---|---|---|---|---|---|
| tom_ball | 5k | 85.0% | 12.2% | 1.8% | 3.2% | 4.1% |
| tom_ball | 9.75k | 79.6% | 11.1% | 0.5% | 0.0% | 10.4% |
| lily_key | 5k | 70.1% | 2.7% | 2.9% | 2.3% | 2.3% |
| lily_key | 9.75k | 82.3% | 2.1% | 2.2% | 3.1% | 1.4% |

None of the 48 sampled generations on these two prompts reproduced the greedy
repetition level; the highest single sampled value was 16.3% (temp 0.7, `tom_ball`,
5k, seed 1339), against greedy's 70.1–85.0%. This pattern held at both checkpoints,
so on this evaluation the looping behaves as a property of the decoding procedure
rather than of the weights.

Repetition rate and text quality did not move together. Temperature 1.0 recorded the
lowest repetition on `tom_ball` at 9.75k (0.5%) while producing malformed word
fragments in the same generations, and on `girl_store` greedy recorded 0.0% while
every sampling strategy recorded 3.6–13.9%. Entity drift and causal inconsistency
appear in generations from all four sampling strategies as well as from greedy
decoding, which is what section 4 quantifies.

---

## 4. Structured error analysis

_Source: `eval/stage5_error_analysis.md`. 9.75k checkpoint, top-k 40 with temperature
1.0, 8 prompts x 5 seeds (1337–1341) = 40 generations, cap 150 tokens. All scoring
rules were written and committed in `eval_error_analysis.py` before any generation was
produced and were not adjusted afterwards._

### Category 1 — entity / name consistency (mechanical)

| Outcome | Count |
|---|---|
| First name recurs | 13/40 (32%) |
| First name changes to another | 5/40 (12%) |
| First name disappears | 7/40 (18%) |
| No proper noun at all | 15/40 (38%) |
| At least one uninitiated new name | 12/40 (30%) |

Of the 25 generations that contained a name, **13 retained the initial name (52%)**.

### Category 2 — causal / story consistency (checklist, applied by reading)

| Checklist item | Flagged |
|---|---|
| Unexplained object appears | 31/40 (78%) |
| Action with no sensible cause | 40/40 (100%) |
| Character appears/vanishes without transition | 38/40 (95%) |
| Ending contradicts something earlier | 24/40 (60%) |
| **At least one of the four** | **40/40 (100%)** |
| Mean issues per generation | 3.33 of 4 |

### Category 3 — completion quality (mechanical)

| Measure | Count |
|---|---|
| Reached `<EOS>` before the cap | 35/40 (88%) |
| Ends on terminal punctuation | 36/40 (90%) |
| Contains invented words | 9/40 (22%) |
| Grammatical (clean end AND no invented words) | 27/40 (68%) |
| Mean 4-gram repetition | 2.0% |

Category 3's measures (88%, 90%, 68%) are substantially stronger than category 2's
flag rates (78–100% of generations flagged) and category 1's name-retention rate
(52% of named generations) on this evaluation.

---

## 5. Limitations

These bound how far the numbers above should be read.

1. **Category 2 was scored by a single automated LLM scorer** (Claude Code, the coding
   agent) against the fixed checklist in `eval_error_analysis.py`. It was not scored by
   human raters, and the scoring has not been validated against human judgment. Per-flag
   evidence phrases are recorded in `eval/causal_scores.json` so each judgment can be
   checked against the text, but no inter-rater agreement was measured.
2. **No human-written baseline was scored.** Real TinyStories text was never run
   through the same causal checklist, so the 100% flag rate has no reference point: it
   is not known what rate the checklist would produce on human-written stories from the
   same corpus.
3. **Category 1's detector produces false positives.** The reported uninitiated-name
   rate is 12/40 (30%). Four of those 12 generations were flagged only on
   sentence-initial words the exclusion list does not cover (`Whatever`, `Inside`, `Of`,
   `Welcome`), putting the estimated true rate near 8/40 (~20%). The reported figure was
   left unchanged because the rule was fixed before scoring.
4. **Small samples.** The prompt set is 8 prompts; the error analysis covers 40
   generations; the greedy baseline is 1 generation per prompt per checkpoint. Several
   per-prompt differences rest on a single generation.
5. **Single seed families throughout.** Training used seed 1337; sampling used
   1337–1341. Results at other seeds were not measured.
6. **Top-k 40 was an evaluation configuration, not an established best strategy.** It
   was selected to hold decoding fixed for the error analysis. The sampling comparison
   recorded top-k 40 and temperature 1.0 within 0.1 percentage points of each other on
   mean repetition (1.5% vs 1.6%), and no strategy was evaluated for overall output
   quality.
7. **Prompt-set coverage.** All 8 prompts are TinyStories-style story openings, so the
   evaluation says nothing about behavior on other text types.

---

## 6. Conclusion

The 1.77M-parameter Transformer achieves official validation perplexity 9.20 and
clearly learns TinyStories-style local language structure. Greedy decoding exposes
severe repetition on some prompts, while sampling largely removes those loops.
Sampling does not eliminate entity drift or causal inconsistencies: in the
40-generation structured analysis, every generation triggered at least one
causal-consistency flag, and 52% of generations containing a name retained the initial
name. Within this evaluation, the model's strongest capability is local language and
surface-form modeling, while maintaining entities and coherent causal state across a
story remains a major limitation — a reading that holds subject to the limitations in
section 5, particularly the single automated scorer and the absence of a human-written
baseline for the causal checklist.
