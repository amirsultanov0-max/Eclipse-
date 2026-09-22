# Seed replication — 20 MB baseline, seed 1339

## sr_base20mb_s1339 — 2026-09-22 20:45

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 10,000 steps this run (through step 10,000), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1339, device `mps`
- data: `data/stage3_train_tokens.npy` — 4,691,267 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 17.5 epochs
- wall clock: 39.2 min (235 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **1.9498** | **2.1972** | **9.0** |
| best val | — | 2.1972 | 9.0 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.1972** | **9.0** |

### Gradient norms (pre-clip, all steps)

- median 0.758, max 3.442 (4.5x median), clipped on 0.4% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.7539 | 4.0739 | 58.8 |
| 500 | 3.8308 | 3.5980 | 36.5 |
| 750 | 3.3807 | 3.2395 | 25.5 |
| 1,000 | 3.0760 | 3.0026 | 20.1 |
| 1,250 | 2.8655 | 2.8418 | 17.1 |
| 1,500 | 2.7233 | 2.7320 | 15.4 |
| 1,750 | 2.6100 | 2.6391 | 14.0 |
| 2,000 | 2.5215 | 2.5774 | 13.2 |
| 2,250 | 2.4632 | 2.5125 | 12.3 |
| 2,500 | 2.4036 | 2.4738 | 11.9 |
| 2,750 | 2.3499 | 2.4412 | 11.5 |
| 3,000 | 2.3188 | 2.4143 | 11.2 |
| 3,250 | 2.2836 | 2.3955 | 11.0 |
| 3,500 | 2.2566 | 2.3735 | 10.7 |
| 3,750 | 2.2240 | 2.3511 | 10.5 |
| 4,000 | 2.1965 | 2.3379 | 10.4 |
| 4,250 | 2.1858 | 2.3197 | 10.2 |
| 4,500 | 2.1635 | 2.3120 | 10.1 |
| 4,750 | 2.1496 | 2.2977 | 10.0 |
| 5,000 | 2.1414 | 2.2863 | 9.8 |
| 5,250 | 2.1175 | 2.2848 | 9.8 |
| 5,500 | 2.1138 | 2.2718 | 9.7 |
| 5,750 | 2.0886 | 2.2656 | 9.6 |
| 6,000 | 2.0765 | 2.2540 | 9.5 |
| 6,250 | 2.0696 | 2.2565 | 9.5 |
| 6,500 | 2.0637 | 2.2506 | 9.5 |
| 6,750 | 2.0488 | 2.2402 | 9.4 |
| 7,000 | 2.0454 | 2.2327 | 9.3 |
| 7,250 | 2.0321 | 2.2285 | 9.3 |
| 7,500 | 2.0228 | 2.2300 | 9.3 |
| 7,750 | 2.0223 | 2.2283 | 9.3 |
| 8,000 | 2.0098 | 2.2157 | 9.2 |
| 8,250 | 2.0015 | 2.2191 | 9.2 |
| 8,500 | 1.9993 | 2.2134 | 9.1 |
| 8,750 | 1.9803 | 2.2088 | 9.1 |
| 9,000 | 1.9812 | 2.2090 | 9.1 |
| 9,250 | 1.9679 | 2.2041 | 9.1 |
| 9,500 | 1.9706 | 2.1998 | 9.0 |
| 9,750 | 1.9570 | 2.2035 | 9.1 |
| 10,000 | 1.9498 | 2.1972 | 9.0 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 1,000 -> 2,000 | 2.5774 | -0.4252 | 13.16 |
| 2,000 -> 3,000 | 2.4143 | -0.1631 | 11.18 |
| 3,000 -> 4,000 | 2.3379 | -0.0764 | 10.36 |
| 4,000 -> 5,000 | 2.2863 | -0.0516 | 9.84 |
| 5,000 -> 6,000 | 2.2540 | -0.0322 | 9.53 |
| 6,000 -> 7,000 | 2.2327 | -0.0214 | 9.32 |
| 7,000 -> 8,000 | 2.2157 | -0.0170 | 9.17 |
| 8,000 -> 9,000 | 2.2090 | -0.0067 | 9.11 |
| 9,000 -> 10,000 | 2.1972 | -0.0118 | 9.00 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Jack. One day she he found a big eating numbers walking and saw something scary! That sounded like Gay, Grace said, Sammy, "Lookue a warm olievence-round in his farm with aian from the middle. She was sad because excited to show it Sp
```

**step 2,000**

```
Once there was a little girl named Lily. She loved to play outside and have fun. 
One day, a little boy named Timmy came to Tom. Tom was scared and didn't know where it looked. He asked his dad why there was from the sun. Her dad said, "No, it door really bad."
```

**step 3,000**

```
Once there was a little girl named Lily. She had a big bin that she liked very much because it was normal and healthy!
One day, Lily's mommy took her a trip to the beach. The beach was so small, it started to leake wet all alone. Lily felt so ashamed that her patch had goals
```

**step 4,000**

```
Once there was a little girl named Sarah. She loved to play outside on the best, but she did not know how.
One day, Sarah heard a strange noise outside. It sounded like the noise was coming from, tensptoplic blinky. Lily was so scared at Stopopt, but
```

**step 5,000**

```
Once there was a little girl named Sarah and her mother went on adventures.
One day of the way, Sarah's mention was playing a lot of charming blue discalk. But when she was playing alone, she spotted a group of seas inside the sea. She was so excited to get reached the big mountains
```

**step 6,000**

```
Once there was a little girl named Jane. Sara was three years old and loved to explore the world around her.
One day, everyone was fighting was unpacking and they decided to go inside it was a cozy way to come. Jane was happy because she always wanted to go outside and collect a safe place.
First
```

**step 7,000**

```
Once there was a little girl named Lily. One night, Lily went to pretending to perform a pretty show about her friends. She was to run and sing.
But then, one day, Lily's friend Flopsy got too noisy. She explained to Lily that if she was good at performing together. Lily didn't
```

**step 8,000**

```
Once there was a little girl named Sally. She was very curious, so she wanted to learn how many help she was, so she could solve any problems because it was. She decided to talk to her way to school. When the shower got his mystery, Sally was so excited to go up and do this was
```

**step 9,000**

```
Once there was a little girl called Lucy. She was only three. Her name was Lucy. Lucy was playing out in the park. She wanted to see the animals, the animals purr. One day, her mommy soon came to the door of the new murrow. There was a scary manb. She screamed
```

**step 10,000**

```
Once there was a little girl who loved to go outdoors to her backyard. Every day she wouldn't feel shy, but she decided to go outside and supply a bag to the backyard. The little girl thought it was very hungry, but the man kept walking until they finally arrived there. The little girl was so
```

Logs: `training.csv`, `per_step.csv`, `samples.txt` in `logs/` · checkpoints: `sr_base20mb_s1339_best.pt` + periodic in `sr_base20mb_s1339/` (both gitignored)
