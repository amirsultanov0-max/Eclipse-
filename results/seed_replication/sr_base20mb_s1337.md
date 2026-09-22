# Seed replication — 20 MB baseline, seed 1337

## sr_base20mb_s1337 — 2026-09-22 16:40

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 10,000 steps this run (through step 10,000), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `mps`
- data: `data/stage3_train_tokens.npy` — 4,691,267 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 17.5 epochs
- wall clock: 32.5 min (195 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **1.9591** | **2.2063** | **9.1** |
| best val | — | 2.2063 | 9.1 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.2063** | **9.1** |

### Gradient norms (pre-clip, all steps)

- median 0.752, max 2.198 (2.9x median), clipped on 0.3% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.8373 | 4.0733 | 58.8 |
| 500 | 3.8371 | 3.6167 | 37.2 |
| 750 | 3.3978 | 3.2547 | 25.9 |
| 1,000 | 3.0954 | 3.0072 | 20.2 |
| 1,250 | 2.8727 | 2.8548 | 17.4 |
| 1,500 | 2.7449 | 2.7466 | 15.6 |
| 1,750 | 2.6299 | 2.6612 | 14.3 |
| 2,000 | 2.5391 | 2.5952 | 13.4 |
| 2,250 | 2.4719 | 2.5432 | 12.7 |
| 2,500 | 2.4121 | 2.4893 | 12.1 |
| 2,750 | 2.3763 | 2.4585 | 11.7 |
| 3,000 | 2.3372 | 2.4387 | 11.5 |
| 3,250 | 2.2951 | 2.4164 | 11.2 |
| 3,500 | 2.2791 | 2.3864 | 10.9 |
| 3,750 | 2.2490 | 2.3656 | 10.7 |
| 4,000 | 2.2166 | 2.3572 | 10.6 |
| 4,250 | 2.1912 | 2.3421 | 10.4 |
| 4,500 | 2.1774 | 2.3256 | 10.2 |
| 4,750 | 2.1620 | 2.3192 | 10.2 |
| 5,000 | 2.1460 | 2.3046 | 10.0 |
| 5,250 | 2.1354 | 2.2974 | 9.9 |
| 5,500 | 2.1118 | 2.2874 | 9.8 |
| 5,750 | 2.0995 | 2.2788 | 9.8 |
| 6,000 | 2.0912 | 2.2692 | 9.7 |
| 6,250 | 2.0796 | 2.2603 | 9.6 |
| 6,500 | 2.0641 | 2.2596 | 9.6 |
| 6,750 | 2.0551 | 2.2561 | 9.5 |
| 7,000 | 2.0515 | 2.2534 | 9.5 |
| 7,250 | 2.0387 | 2.2478 | 9.5 |
| 7,500 | 2.0346 | 2.2413 | 9.4 |
| 7,750 | 2.0238 | 2.2415 | 9.4 |
| 8,000 | 2.0148 | 2.2313 | 9.3 |
| 8,250 | 2.0115 | 2.2249 | 9.3 |
| 8,500 | 2.0010 | 2.2269 | 9.3 |
| 8,750 | 1.9912 | 2.2236 | 9.2 |
| 9,000 | 1.9858 | 2.2173 | 9.2 |
| 9,250 | 1.9872 | 2.2154 | 9.2 |
| 9,500 | 1.9709 | 2.2131 | 9.1 |
| 9,750 | 1.9694 | 2.2080 | 9.1 |
| 10,000 | 1.9591 | 2.2063 | 9.1 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 1,000 -> 2,000 | 2.5952 | -0.4120 | 13.40 |
| 2,000 -> 3,000 | 2.4387 | -0.1565 | 11.46 |
| 3,000 -> 4,000 | 2.3572 | -0.0815 | 10.56 |
| 4,000 -> 5,000 | 2.3046 | -0.0526 | 10.02 |
| 5,000 -> 6,000 | 2.2692 | -0.0355 | 9.67 |
| 6,000 -> 7,000 | 2.2534 | -0.0158 | 9.52 |
| 7,000 -> 8,000 | 2.2313 | -0.0221 | 9.31 |
| 8,000 -> 9,000 | 2.2173 | -0.0139 | 9.18 |
| 9,000 -> 10,000 | 2.2063 | -0.0110 | 9.08 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Lucy who loved to play outside and dad. Jane was very upset and say hello. One day, Jill got to pick it up and as she came to her mom opened arring the rain. Daisy stretched her mom and started into a sparkly piece of the hill.
She had a great
```

**step 2,000**

```
Once there was a little girl named Sally. She was very patiently. Her mom decided to go home.
When she closed her eyes while she went down a toy. When she opened her eyes, where she saw was feeling.
Her mom was so sad, but she kept asking for help. Suddenly, she noticed that
```

**step 3,000**

```
Once there was a little girl named Lily. Lily was very excited, but she was looking for a new toy. She kicked it, and suddenly started to act loudly.
Lily felt sad and started to complain. She asked her mom to applaud Their parents. She asked her, "Please don't go where my
```

**step 4,000**

```
Once there was a little girl named Lily. She loved to play and the toy in the house. One day, Lily's mom asked her to take the car outside and see an frosting ice cream. 
"Can I go inside and build some delicious strawberries with it!" Lily exclaimed. 
Her mom obser
```

**step 5,000**

```
Once there was a little girl named Lily. She loved her swing and it was very cold. One day, she decided to ripped the zipper to a new playomplace. She sobbed herself all over her swing but couldn't get it. Then, her parents watched her swing back and stopped playing.
Lily was
```

**step 6,000**

```
Once there was a little girl called Amy. She was very obedient and always recorded. One day the little girl came to her house. She was so excited to show her mom the bracelet.
She wanted to show her mom a statue to her. Her mom was feeling jealous and asked her some kids what was doing.

```

**step 7,000**

```
Once there was a little girl named Lucy. She went to the park and saw that vegetables were so good in the park. She watched as the vegetables and saw a fruit. After a while she reached the top of a tree, she paused to get a closer look. Suddenly, something strange happened. Lucy felt surprised and she
```

**step 8,000**

```
Once there was a little girl named Lola. Rose wanted to always keep things clean. At first, she asked her mommy to help her serve her in dinner. Her mommy said it was so important to pack people's toys and supply kids with the dishes they would be organized.
One day, Lola's brother Tommy noticed
```

**step 9,000**

```
Once there was a little girl who was very curious. She wanted to have something for her first time. But it was her and scared. She tried to explode something delicious.
But she couldn't seem to give up. As she tried her best and shouted out to herself, 'No, it is too big.
```

**step 10,000**

```
Once there was a little girl who loved to eat rice. One day she thought, something magical happened. So she took a clean cup of rice to wash it. 
As she stirred the rice, the rice began to shrink. Suddenly, she had an idea! She was able to quarrel over but she realized
```

Logs: `training.csv`, `per_step.csv`, `samples.txt` in `logs/` · checkpoints: `sr_base20mb_s1337_best.pt` + periodic in `sr_base20mb_s1337/` (both gitignored)
