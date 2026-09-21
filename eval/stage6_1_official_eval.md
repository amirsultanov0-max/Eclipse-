
---

## Official validation set — 2026-09-21 22:41

Read-only evaluation of `transformer_10k_best.pt` (full file) (step 9,750) on `data/tinystories_valid.txt`, reserved and untouched since Stage 1. One deterministic pass, every token scored exactly once — no sampling, no repeats.

- 21,990 stories -> 4,797,239 tokens (BPE, one `<EOS>` per story, same construction as `prepare_data.py`)
- 9,369 non-overlapping windows of 512; 311 tokens dropped in the final partial window
- **4,787,559 tokens scored** (99.80% of the file; each window's first token is an input only, never a target)
- cross-check at stride 512 with 513-token windows, which scores every token: 4,796,928 tokens, loss 2.2197 (ppl 9.20) — the windowing choice is worth 0.0000 nats

| model | loss (nats) | perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000.00 |
| unigram (no context) | 6.0360 | 418.24 |
| Stage 3 trained bigram (1-token context) | 4.6542 | 105.02 |
| count-based bigram floor (add-0.01) | 3.7686 | 43.32 |
| **5k transformer, step 5,000 (recorded)** | **2.3236** | **10.21** |
| **this transformer, step 9,750 (512-token context)** | **2.2197** | **9.20** |

Every row is measured on this same file, over the same token pairs. The Stage 3 bigram was re-scored from its own checkpoint rather than carried over. Bigram and unigram counts come from the training stream — only the evaluation data changed.

This is an architectural comparison, not a like-for-like one: the bigram baselines see 1 token of context, the transformer up to 512.

### Loss by position in the window

| target positions | context available | loss | perplexity |
|---|---|---|---|
| 1-16 | 1-16 tokens | 2.9134 | 18.42 |
| 17-64 | 17-64 tokens | 2.3311 | 10.29 |
| 65-128 | 65-128 tokens | 2.1998 | 9.02 |
| 129-256 | 129-256 tokens | 2.1785 | 8.83 |
| 257-511 | 257-511 tokens | 2.1810 | 8.85 |

Each window starts cold, so early positions are predicted from almost nothing. This is what the reserved-file number costs relative to a sliding window, and it is also a direct measure of what context is worth.

---

## Official validation set — 2026-09-21 22:42

Read-only evaluation of `transformer_10k_best.pt` (clean subset) (step 9,750) on `data/tinystories_valid.txt`, reserved and untouched since Stage 1. One deterministic pass, every token scored exactly once — no sampling, no repeats.

- 21,890 stories -> 4,766,371 tokens (BPE, one `<EOS>` per story, same construction as `prepare_data.py`)
- 9,309 non-overlapping windows of 512; 163 tokens dropped in the final partial window
- **4,756,899 tokens scored** (99.80% of the file; each window's first token is an input only, never a target)
- cross-check at stride 512 with 513-token windows, which scores every token: 4,766,208 tokens, loss 2.2223 (ppl 9.23) — the windowing choice is worth 0.0000 nats

| model | loss (nats) | perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000.00 |
| unigram (no context) | 6.0369 | 418.60 |
| Stage 3 trained bigram (1-token context) | 4.6558 | 105.20 |
| count-based bigram floor (add-0.01) | 3.7699 | 43.38 |
| **this transformer, step 9,750 (512-token context)** | **2.2223** | **9.23** |

Every row is measured on this same file, over the same token pairs. The Stage 3 bigram was re-scored from its own checkpoint rather than carried over. Bigram and unigram counts come from the training stream — only the evaluation data changed.

This is an architectural comparison, not a like-for-like one: the bigram baselines see 1 token of context, the transformer up to 512.

### Loss by position in the window

| target positions | context available | loss | perplexity |
|---|---|---|---|
| 1-16 | 1-16 tokens | 2.9193 | 18.53 |
| 17-64 | 17-64 tokens | 2.3336 | 10.31 |
| 65-128 | 65-128 tokens | 2.2037 | 9.06 |
| 129-256 | 129-256 tokens | 2.1773 | 8.82 |
| 257-511 | 257-511 tokens | 2.1849 | 8.89 |

Each window starts cold, so early positions are predicted from almost nothing. This is what the reserved-file number costs relative to a sliding window, and it is also a direct measure of what context is worth.

---

## Official validation set — 2026-09-21 22:44

Read-only evaluation of `transformer_6_1_best.pt` (full file) (step 9,750) on `data/tinystories_valid.txt`, reserved and untouched since Stage 1. One deterministic pass, every token scored exactly once — no sampling, no repeats.

- 21,990 stories -> 4,797,239 tokens (BPE, one `<EOS>` per story, same construction as `prepare_data.py`)
- 9,369 non-overlapping windows of 512; 311 tokens dropped in the final partial window
- **4,787,559 tokens scored** (99.80% of the file; each window's first token is an input only, never a target)
- cross-check at stride 512 with 513-token windows, which scores every token: 4,796,928 tokens, loss 2.1097 (ppl 8.25) — the windowing choice is worth 0.0000 nats

| model | loss (nats) | perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000.00 |
| unigram (no context) | 6.0360 | 418.24 |
| Stage 3 trained bigram (1-token context) | 4.6542 | 105.02 |
| count-based bigram floor (add-0.01) | 3.7686 | 43.32 |
| **5k transformer, step 5,000 (recorded)** | **2.3236** | **10.21** |
| **this transformer, step 9,750 (512-token context)** | **2.1097** | **8.25** |

Every row is measured on this same file, over the same token pairs. The Stage 3 bigram was re-scored from its own checkpoint rather than carried over. Bigram and unigram counts come from the training stream — only the evaluation data changed.

This is an architectural comparison, not a like-for-like one: the bigram baselines see 1 token of context, the transformer up to 512.

### Loss by position in the window

| target positions | context available | loss | perplexity |
|---|---|---|---|
| 1-16 | 1-16 tokens | 2.8432 | 17.17 |
| 17-64 | 17-64 tokens | 2.2376 | 9.37 |
| 65-128 | 65-128 tokens | 2.0914 | 8.10 |
| 129-256 | 129-256 tokens | 2.0633 | 7.87 |
| 257-511 | 257-511 tokens | 2.0675 | 7.90 |

Each window starts cold, so early positions are predicted from almost nothing. This is what the reserved-file number costs relative to a sliding window, and it is also a direct measure of what context is worth.

---

## Official validation set — 2026-09-21 22:45

Read-only evaluation of `transformer_6_1_best.pt` (clean subset) (step 9,750) on `data/tinystories_valid.txt`, reserved and untouched since Stage 1. One deterministic pass, every token scored exactly once — no sampling, no repeats.

- 21,890 stories -> 4,766,371 tokens (BPE, one `<EOS>` per story, same construction as `prepare_data.py`)
- 9,309 non-overlapping windows of 512; 163 tokens dropped in the final partial window
- **4,756,899 tokens scored** (99.80% of the file; each window's first token is an input only, never a target)
- cross-check at stride 512 with 513-token windows, which scores every token: 4,766,208 tokens, loss 2.1110 (ppl 8.26) — the windowing choice is worth 0.0000 nats

| model | loss (nats) | perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000.00 |
| unigram (no context) | 6.0369 | 418.60 |
| Stage 3 trained bigram (1-token context) | 4.6558 | 105.20 |
| count-based bigram floor (add-0.01) | 3.7699 | 43.38 |
| **this transformer, step 9,750 (512-token context)** | **2.1110** | **8.26** |

Every row is measured on this same file, over the same token pairs. The Stage 3 bigram was re-scored from its own checkpoint rather than carried over. Bigram and unigram counts come from the training stream — only the evaluation data changed.

This is an architectural comparison, not a like-for-like one: the bigram baselines see 1 token of context, the transformer up to 512.

### Loss by position in the window

| target positions | context available | loss | perplexity |
|---|---|---|---|
| 1-16 | 1-16 tokens | 2.8509 | 17.30 |
| 17-64 | 17-64 tokens | 2.2352 | 9.35 |
| 65-128 | 65-128 tokens | 2.0930 | 8.11 |
| 129-256 | 129-256 tokens | 2.0617 | 7.86 |
| 257-511 | 257-511 tokens | 2.0705 | 7.93 |

Each window starts cold, so early positions are predicted from almost nothing. This is what the reserved-file number costs relative to a sliding window, and it is also a direct measure of what context is worth.
