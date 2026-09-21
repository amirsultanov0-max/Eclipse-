# Stage 4 — transformer results

## transformer_10k — 2026-09-21 19:51

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 5,000 steps this run (through step 10,000), resumed from `checkpoints/transformer_best.pt` at step 5,000, batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `mps`
- data: Stage 3's 95/5 split — 4,691,267 train tokens, validation from the monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 17.5 epochs
- wall clock: 21.4 min (257 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **1.9642** | **2.2074** | **9.1** |
| best val | — | 2.2031 | 9.1 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.2074** | **9.1** |

### Gradient norms (pre-clip, all steps)

- median 0.730, max 0.906 (1.2x median), clipped on 0.0% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 5,250 | 2.1360 | 2.2945 | 9.9 |
| 5,500 | 2.1117 | 2.2862 | 9.8 |
| 5,750 | 2.0999 | 2.2819 | 9.8 |
| 6,000 | 2.0927 | 2.2693 | 9.7 |
| 6,250 | 2.0782 | 2.2643 | 9.6 |
| 6,500 | 2.0704 | 2.2602 | 9.6 |
| 6,750 | 2.0481 | 2.2514 | 9.5 |
| 7,000 | 2.0510 | 2.2481 | 9.5 |
| 7,250 | 2.0383 | 2.2428 | 9.4 |
| 7,500 | 2.0367 | 2.2371 | 9.4 |
| 7,750 | 2.0240 | 2.2371 | 9.4 |
| 8,000 | 2.0183 | 2.2294 | 9.3 |
| 8,250 | 2.0054 | 2.2319 | 9.3 |
| 8,500 | 2.0018 | 2.2271 | 9.3 |
| 8,750 | 1.9925 | 2.2171 | 9.2 |
| 9,000 | 1.9897 | 2.2210 | 9.2 |
| 9,250 | 1.9797 | 2.2189 | 9.2 |
| 9,500 | 1.9764 | 2.2127 | 9.1 |
| 9,750 | 1.9705 | 2.2031 | 9.1 |
| 10,000 | 1.9642 | 2.2074 | 9.1 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 6,000 -> 7,000 | 2.2481 | -0.0212 | 9.47 |
| 7,000 -> 8,000 | 2.2294 | -0.0186 | 9.29 |
| 8,000 -> 9,000 | 2.2210 | -0.0084 | 9.22 |
| 9,000 -> 10,000 | 2.2074 | -0.0136 | 9.09 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 6,000**

```
Once there was a little girl named Polly. One day Polly was playing outside when she saw something strange. It was a voice at her. Peter was scared but he wanted to it. He asked himself like that reindeer. He said it was dangerous. Leakes noticed that goose was chasing him. She smiled and said "Please,
```

**step 7,000**

```
Once there was a little girl who loved to go to the zoo. One day it was pale, and she wanted to go on an adventure.
When she got to the zoo, the little girl knocked and said, "Oh no! Let's go!" But the little girl refused to do the her wife's learn. She
```

**step 8,000**

```
Once there was a little girl called Bora. She was 3 years old and loved to explore. One day she decided to get a test at all the entire world. 
“Mom, my sister.'Look. It is amazing!" asked Penn't believe.
Mom smiled. "That's a
```

**step 9,000**

```
Once there was a little girl who was very important. She had a new cat that she loved very much. Every day she brought everything like ice cream. 
One day she was walking down the street, pointing at the beautiful veluty storegler as joy as it was out of orange.
The girl
```

**step 10,000**

```
Once there was a little girl called Jane. One day, she wanted to find her mom. She said, "Mom, what are you doing?"
Her mom explained, "I'm trying to find the hold so you have a key to keep it safe. Do you understand?"
Jane nodded and turned on her mom's
```

Logs: `transformer_training_10k.csv`, `transformer_grad_norms_10k.csv`, `transformer_samples_10k.txt` in `logs/` · checkpoints: `transformer_10k_best.pt` + periodic in `checkpoints/` (both gitignored)
