
---

## Official validation set — 2026-09-23 20:59

Read-only evaluation of `st62_d352_s1339_best.pt` (clean subset) (step 10,000) on `data/tinystories_valid.txt`, reserved and untouched since Stage 1. One deterministic pass, every token scored exactly once — no sampling, no repeats.

- 21,890 stories -> 4,766,371 tokens (BPE, one `<EOS>` per story, same construction as `prepare_data.py`)
- 9,309 non-overlapping windows of 512; 163 tokens dropped in the final partial window
- **4,756,899 tokens scored** (99.80% of the file; each window's first token is an input only, never a target)
- cross-check at stride 512 with 513-token windows, which scores every token: 4,766,208 tokens, loss 1.9207 (ppl 6.83) — the windowing choice is worth 0.0001 nats

| model | loss (nats) | perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000.00 |
| unigram (no context) | 6.0369 | 418.60 |
| Stage 3 trained bigram (1-token context) | 4.6558 | 105.20 |
| count-based bigram floor (add-0.01) | 3.7699 | 43.38 |
| **this transformer, step 10,000 (512-token context)** | **1.9208** | **6.83** |

Every row is measured on this same file, over the same token pairs. The Stage 3 bigram was re-scored from its own checkpoint rather than carried over. Bigram and unigram counts come from the training stream — only the evaluation data changed.

This is an architectural comparison, not a like-for-like one: the bigram baselines see 1 token of context, the transformer up to 512.

### Loss by position in the window

| target positions | context available | loss | perplexity |
|---|---|---|---|
| 1-16 | 1-16 tokens | 2.7238 | 15.24 |
| 17-64 | 17-64 tokens | 2.0569 | 7.82 |
| 65-128 | 65-128 tokens | 1.9041 | 6.71 |
| 129-256 | 129-256 tokens | 1.8686 | 6.48 |
| 257-511 | 257-511 tokens | 1.8751 | 6.52 |

Each window starts cold, so early positions are predicted from almost nothing. This is what the reserved-file number costs relative to a sliding window, and it is also a direct measure of what context is worth.
