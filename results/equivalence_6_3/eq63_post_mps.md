# Stage 6.3 equivalence run

## eq63_20260925120839_post_mps — 2026-09-25 12:28

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 1,001 steps this run (through step 1,001), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `mps`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 8,200,192 tokens = 0.2 epochs
- wall clock: 3.9 min (233 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 1,001)** | **3.0338** | **2.9942** | **20.0** |
| best val | — | 2.9908 | 19.9 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.9942** | **20.0** |

### Gradient norms (pre-clip, all steps)

- median 0.661, max 45.745 (69.2x median), clipped on 2.8% of steps
- **1 step(s) above 10.0x the median** — step 17 (45.7). This is the pattern the lr=3e-3 sweep run showed, where clipping hid it from the loss.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.7895 | 4.0492 | 57.4 |
| 500 | 3.8333 | 3.6063 | 36.8 |
| 750 | 3.4187 | 3.2380 | 25.5 |
| 1,000 | 3.1192 | 2.9908 | 19.9 |
| 1,001 | 3.0338 | 2.9942 | 20.0 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Lucy. She was three years old and a little girl named Bill. One day, Jack heard a voice calling him. She saw a round bow mintion on its face and tried to admire it. 
The little girl felt bad and clapped on her crumated. She felt
```

Logs: `eq63_20260925120839_post_mps_training.csv`, `eq63_20260925120839_post_mps_grad.csv`, `eq63_20260925120839_post_mps_samples.txt` in `logs/` · checkpoints: `eq63_20260925120839_post_mps_best.pt` + periodic in `ckpt_post_mps/` (both gitignored)
