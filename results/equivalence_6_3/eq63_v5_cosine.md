# Stage 6.3 equivalence run

## eq63_20260925120839_v5_cosine — 2026-09-25 12:35

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 20 steps this run (through step 20), batch 16, lr 0.001 peak (linear warm-up over steps 1-1,000, then cosine decay to 0.0001 at step 10,000), AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `cpu`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 163,840 tokens = 0.0 epochs
- wall clock: 0.3 min (783 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 20)** | **8.2651** | **8.1603** | **3499.2** |
| best val | — | 8.1603 | 3499.2 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **8.1603** | **3499.2** |

### Gradient norms (pre-clip, all steps)

- median 2.093, max 2.243 (1.1x median), clipped on 100.0% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 20 | 8.2651 | 8.1603 | 3499.2 |
Logs: `eq63_20260925120839_v5_cosine_training.csv`, `eq63_20260925120839_v5_cosine_grad.csv`, `eq63_20260925120839_v5_cosine_samples.txt` in `logs/` · checkpoints: `eq63_20260925120839_v5_cosine_best.pt` + periodic in `ckpt_v5_cosine/` (both gitignored)
