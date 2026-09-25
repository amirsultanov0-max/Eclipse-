# Stage 6.3 equivalence run

## eq63_20260925120839_post_cpu — 2026-09-25 12:20

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 1,001 steps this run (through step 1,001), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `cpu`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 8,200,192 tokens = 0.2 epochs
- wall clock: 6.5 min (389 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 1,001)** | **2.8966** | **2.9900** | **19.9** |
| best val | — | 2.9900 | 19.9 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.9900** | **19.9** |

### Gradient norms (pre-clip, all steps)

- median 0.681, max 2.275 (3.3x median), clipped on 3.7% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.8235 | 4.0514 | 57.5 |
| 500 | 3.8212 | 3.5810 | 35.9 |
| 750 | 3.3889 | 3.2143 | 24.9 |
| 1,000 | 3.1039 | 2.9912 | 19.9 |
| 1,001 | 2.8966 | 2.9900 | 19.9 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl who wanted to walk. She looked around and saw a butterfly and bright pretty bird called her name bench. He wanted to pray and see why were happened, and closer to catch. He asked it to help but it thoughtled a terrible idea.
Suddenly one was amazing, a little bunny or wr
```

Logs: `eq63_20260925120839_post_cpu_training.csv`, `eq63_20260925120839_post_cpu_grad.csv`, `eq63_20260925120839_post_cpu_samples.txt` in `logs/` · checkpoints: `eq63_20260925120839_post_cpu_best.pt` + periodic in `ckpt_post_cpu/` (both gitignored)
