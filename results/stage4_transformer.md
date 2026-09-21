# Stage 4 — transformer results

## transformer — 2026-09-21 19:18

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 5,000 steps, batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `mps`
- data: Stage 3's 95/5 split — 4,691,267 train tokens, validation from the monitoring split (`tinystories_valid.txt` untouched)
- saw 40,960,000 tokens = 8.7 epochs
- wall clock: 16.7 min (200 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 5,000)** | **2.1490** | **2.3079** | **10.1** |
| best val | — | 2.3079 | 10.1 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.3079** | **10.1** |

### Gradient norms (pre-clip, all steps)

- median 0.791, max 2.198 (2.8x median), clipped on 0.7% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.8373 | 4.0733 | 58.8 |
| 500 | 3.8372 | 3.6169 | 37.2 |
| 750 | 3.3977 | 3.2527 | 25.9 |
| 1,000 | 3.0950 | 3.0068 | 20.2 |
| 1,250 | 2.8829 | 2.8560 | 17.4 |
| 1,500 | 2.7345 | 2.7408 | 15.5 |
| 1,750 | 2.6350 | 2.6625 | 14.3 |
| 2,000 | 2.5381 | 2.5939 | 13.4 |
| 2,250 | 2.4755 | 2.5368 | 12.6 |
| 2,500 | 2.4211 | 2.5063 | 12.3 |
| 2,750 | 2.3696 | 2.4618 | 11.7 |
| 3,000 | 2.3336 | 2.4379 | 11.4 |
| 3,250 | 2.3028 | 2.4085 | 11.1 |
| 3,500 | 2.2656 | 2.3906 | 10.9 |
| 3,750 | 2.2494 | 2.3718 | 10.7 |
| 4,000 | 2.2145 | 2.3614 | 10.6 |
| 4,250 | 2.2000 | 2.3409 | 10.4 |
| 4,500 | 2.1854 | 2.3258 | 10.2 |
| 4,750 | 2.1612 | 2.3137 | 10.1 |
| 5,000 | 2.1490 | 2.3079 | 10.1 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Lily. She loved to explore a long, and her ball. One day, she asked her mom if she asked her mummy if she they go back. They packed the decide to the zoo forever.
As they were tired, the park, Lily saw something strange. She was scared and smiled
```

**step 2,000**

```
Once there was a little girl named it. She was very excited, because she had lots of hid.
One day, when she saw a big grill in plan. "Please there?" she asked.
"Why don't you help me, Mr. Frollow me when you can drive me?" bent down from
```

**step 3,000**

```
Once there was a little girl called Jack. He wanted to go outside to play with his friends. But he didn't know why it was too heavy.  her friend Jack didn't have to go off, so he reached up and examine it again. Eventually, it was dark in the dark. The sky was so big
```

**step 4,000**

```
Once there was a little girl named Lily who wanted to help her pet dolls. 
Lily was a kind and asked her mommy if they could made her dirty go. Lily was so excited to help her watch all the animals. 
When it was time to go home, Lily's mommy put on their band-aid and
```

**step 5,000**

```
Once there was a little girl called Lucy. She was so happy, she had an amazing day.
Sophie prepared the way out of it was always playing with loud noises. Every day, Lucy and her parents would look out beautiful animals came across different streets. 
The day was very exciting that Lucy had never
```

Logs: `transformer_training.csv`, `transformer_grad_norms.csv`, `transformer_samples.txt` in `logs/` · checkpoints: `transformer_best.pt` + periodic in `checkpoints/` (both gitignored)

---

## Official validation set — 2026-09-21 19:26

Read-only evaluation of `checkpoints/transformer_best.pt` (step 5,000) on `data/tinystories_valid.txt`, reserved and untouched since Stage 1. One deterministic pass, every token scored exactly once — no sampling, no repeats.

- 21,990 stories -> 4,797,239 tokens (BPE, one `<EOS>` per story, same construction as `prepare_data.py`)
- 9,369 non-overlapping windows of 512; 311 tokens dropped in the final partial window
- **4,787,559 tokens scored** (99.80% of the file; each window's first token is an input only, never a target)
- cross-check at stride 512 with 513-token windows, which scores every token: 4,796,928 tokens, loss 2.3236 (ppl 10.21) — the windowing choice is worth 0.0000 nats

| model | loss (nats) | perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000.00 |
| unigram (no context) | 6.0360 | 418.24 |
| Stage 3 trained bigram (1-token context) | 4.6542 | 105.02 |
| count-based bigram floor (add-0.01) | 3.7686 | 43.32 |
| **this transformer (512-token context)** | **2.3236** | **10.21** |

Every row is measured on this same file, over the same token pairs. The Stage 3 bigram was re-scored from its own checkpoint rather than carried over. Bigram and unigram counts come from the training stream — only the evaluation data changed.

This is an architectural comparison, not a like-for-like one: the bigram baselines see 1 token of context, the transformer up to 512.

### Loss by position in the window

| target positions | context available | loss | perplexity |
|---|---|---|---|
| 1-16 | 1-16 tokens | 3.0036 | 20.16 |
| 17-64 | 17-64 tokens | 2.4373 | 11.44 |
| 65-128 | 65-128 tokens | 2.3061 | 10.04 |
| 129-256 | 129-256 tokens | 2.2797 | 9.77 |
| 257-511 | 257-511 tokens | 2.2859 | 9.83 |

Each window starts cold, so early positions are predicted from almost nothing. This is what the reserved-file number costs relative to a sliding window, and it is also a direct measure of what context is worth.
