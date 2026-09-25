# Stage 6.3 equivalence run

## eq63_20260925120839_treat_cpu — 2026-09-25 12:35

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 1,001 steps this run (through step 1,001), batch 16, lr 0.001 peak (linear warm-up over steps 1-1,000, then cosine decay to 0.0001 at step 10,000), AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `cpu`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 8,200,192 tokens = 0.2 epochs
- wall clock: 6.8 min (406 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 1,001)** | **3.2468** | **3.3654** | **28.9** |
| best val | — | 3.3530 | 28.6 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **3.3654** | **28.9** |

### Gradient norms (pre-clip, all steps)

- median 0.986, max 2.719 (2.8x median), clipped on 48.5% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 6.8286 | 5.1676 | 175.5 |
| 500 | 4.5207 | 4.1688 | 64.6 |
| 750 | 3.9571 | 3.7248 | 41.5 |
| 1,000 | 3.5346 | 3.3530 | 28.6 |
| 1,001 | 3.2468 | 3.3654 | 28.9 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Lily who wanted to have, she bark in the park. She went and called her with her mommy. The woman said, "We'm sorry for your family."
Her mommy asked it thought it was veryebled out. She asked the boy if she could do.
Tom or wr
```

Logs: `eq63_20260925120839_treat_cpu_training.csv`, `eq63_20260925120839_treat_cpu_grad.csv`, `eq63_20260925120839_treat_cpu_samples.txt` in `logs/` · checkpoints: `eq63_20260925120839_treat_cpu_best.pt` + periodic in `ckpt_treat_cpu/` (both gitignored)
