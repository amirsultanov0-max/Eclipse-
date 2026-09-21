# Stage 3 — bigram model results

One section per training run, appended automatically by train.py.

**Methodology note (2026-09-21):** the add-k smoothing constant for the count-based
bigram baseline is chosen on a 5% slice held out of the *training* stream. An earlier
version picked it by validation score, which tuned that baseline against the data it is
then measured on. Both methods land on k=0.01, so the numbers below are unchanged
(train 3.5840 / val 3.7715); a wider grid (0.001–1.0) confirms k=0.01 is a genuine
minimum rather than the edge of the old grid.

## batch64 — 2026-09-21 17:50

- config: 5,000 steps, batch 64, lr 0.001, plain Adam, zero-init, seed 1337, device `mps`
- data: 4,691,267 train tokens / 254,283 val tokens (95/5 split by story, BPE vocab 4,000)
- saw 320,000 pairs = 6.8% of one epoch
- wall clock: 63.8s (12.77 ms/step)

| loss (nats) | train | val |
|---|---|---|
| uniform, log(4000) | 8.2940 | 8.2940 |
| unigram (no context) | 6.0243 | 6.0253 |
| count-based bigram (add-0.01, k chosen on a held-out training slice) | 3.5840 | 3.7715 |
| count-based bigram (unsmoothed) | 3.5561 | inf |
| **trained model** | **6.4227** | **6.4285** |
| trained model perplexity | 615.7 | 619.2 |

Log: `logs/stage3_batch64.csv` · checkpoint: `checkpoints/stage3_batch64.pt` (both gitignored)

## batch512 — 2026-09-21 17:51

- config: 5,000 steps, batch 512, lr 0.001, plain Adam, zero-init, seed 1337, device `mps`
- data: 4,691,267 train tokens / 254,283 val tokens (95/5 split by story, BPE vocab 4,000)
- saw 2,560,000 pairs = 54.6% of one epoch
- wall clock: 69.3s (13.85 ms/step)

| loss (nats) | train | val |
|---|---|---|
| uniform, log(4000) | 8.2940 | 8.2940 |
| unigram (no context) | 6.0243 | 6.0253 |
| count-based bigram (add-0.01, k chosen on a held-out training slice) | 3.5840 | 3.7715 |
| count-based bigram (unsmoothed) | 3.5561 | inf |
| **trained model** | **5.2875** | **5.3508** |
| trained model perplexity | 197.8 | 210.8 |

Log: `logs/stage3_batch512.csv` · checkpoint: `checkpoints/stage3_batch512.pt` (both gitignored)

## batch64_lr0.1 — 2026-09-21 17:52

- config: 5,000 steps, batch 64, lr 0.1, plain Adam, zero-init, seed 1337, device `mps`
- data: 4,691,267 train tokens / 254,283 val tokens (95/5 split by story, BPE vocab 4,000)
- saw 320,000 pairs = 6.8% of one epoch
- wall clock: 64.0s (12.80 ms/step)

| loss (nats) | train | val |
|---|---|---|
| uniform, log(4000) | 8.2940 | 8.2940 |
| unigram (no context) | 6.0243 | 6.0253 |
| count-based bigram (add-0.01, k chosen on a held-out training slice) | 3.5840 | 3.7715 |
| count-based bigram (unsmoothed) | 3.5561 | inf |
| **trained model** | **4.6122** | **4.6354** |
| trained model perplexity | 100.7 | 103.1 |

Log: `logs/stage3_batch64_lr0.1.csv` · checkpoint: `checkpoints/stage3_batch64_lr0.1.pt` (both gitignored)
