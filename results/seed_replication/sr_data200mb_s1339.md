# Seed replication — 200 MB (6.1), seed 1339

## sr_data200mb_s1339 — 2026-09-22 21:28

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 10,000 steps this run (through step 10,000), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1339, device `mps`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 1.7 epochs
- wall clock: 39.6 min (237 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **2.0741** | **2.0803** | **8.0** |
| best val | — | 2.0803 | 8.0 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.0803** | **8.0** |

### Gradient norms (pre-clip, all steps)

- median 0.696, max 2.728 (3.9x median), clipped on 0.3% of steps
- No step exceeded 10.0x the median, so nothing resembling the lr=3e-3 spike pattern (38.4x at step 12) appeared.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.7499 | 4.0588 | 57.9 |
| 500 | 3.8532 | 3.6028 | 36.7 |
| 750 | 3.4137 | 3.2271 | 25.2 |
| 1,000 | 3.0999 | 2.9861 | 19.8 |
| 1,250 | 2.8997 | 2.8258 | 16.9 |
| 1,500 | 2.7653 | 2.7042 | 14.9 |
| 1,750 | 2.6617 | 2.6164 | 13.7 |
| 2,000 | 2.5873 | 2.5444 | 12.7 |
| 2,250 | 2.5292 | 2.4931 | 12.1 |
| 2,500 | 2.4667 | 2.4415 | 11.5 |
| 2,750 | 2.4251 | 2.4013 | 11.0 |
| 3,000 | 2.3916 | 2.3752 | 10.8 |
| 3,250 | 2.3581 | 2.3412 | 10.4 |
| 3,500 | 2.3356 | 2.3158 | 10.1 |
| 3,750 | 2.3168 | 2.2996 | 10.0 |
| 4,000 | 2.2872 | 2.2819 | 9.8 |
| 4,250 | 2.2722 | 2.2628 | 9.6 |
| 4,500 | 2.2588 | 2.2463 | 9.5 |
| 4,750 | 2.2452 | 2.2354 | 9.4 |
| 5,000 | 2.2207 | 2.2247 | 9.3 |
| 5,250 | 2.2142 | 2.2075 | 9.1 |
| 5,500 | 2.2056 | 2.1998 | 9.0 |
| 5,750 | 2.1834 | 2.1931 | 9.0 |
| 6,000 | 2.1750 | 2.1816 | 8.9 |
| 6,250 | 2.1684 | 2.1744 | 8.8 |
| 6,500 | 2.1484 | 2.1596 | 8.7 |
| 6,750 | 2.1521 | 2.1598 | 8.7 |
| 7,000 | 2.1401 | 2.1477 | 8.6 |
| 7,250 | 2.1448 | 2.1441 | 8.5 |
| 7,500 | 2.1364 | 2.1312 | 8.4 |
| 7,750 | 2.1220 | 2.1300 | 8.4 |
| 8,000 | 2.1245 | 2.1240 | 8.4 |
| 8,250 | 2.1147 | 2.1159 | 8.3 |
| 8,500 | 2.1121 | 2.1093 | 8.2 |
| 8,750 | 2.1007 | 2.1042 | 8.2 |
| 9,000 | 2.0939 | 2.1000 | 8.2 |
| 9,250 | 2.0948 | 2.0960 | 8.1 |
| 9,500 | 2.0846 | 2.0890 | 8.1 |
| 9,750 | 2.0763 | 2.0862 | 8.1 |
| 10,000 | 2.0741 | 2.0803 | 8.0 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 1,000 -> 2,000 | 2.5444 | -0.4417 | 12.74 |
| 2,000 -> 3,000 | 2.3752 | -0.1692 | 10.75 |
| 3,000 -> 4,000 | 2.2819 | -0.0933 | 9.80 |
| 4,000 -> 5,000 | 2.2247 | -0.0572 | 9.25 |
| 5,000 -> 6,000 | 2.1816 | -0.0430 | 8.86 |
| 6,000 -> 7,000 | 2.1477 | -0.0340 | 8.56 |
| 7,000 -> 8,000 | 2.1240 | -0.0237 | 8.36 |
| 8,000 -> 9,000 | 2.1000 | -0.0239 | 8.17 |
| 9,000 -> 10,000 | 2.0803 | -0.0197 | 8.01 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Lily. One day she he found a big arm and the loved a cookie. Lily wanted to look over wet and shrugged. They wore it on the shelf. While they got toysThey sat down in the farm. Lily was very sad because they wanted to do it. The ant was the
```

**step 2,000**

```
Once there was a little girl named Lily. She loved to play. One day, Lily's mom brought water to a tea party. "Here we go more left!" she said.
Her mom smiled and said, "Look, Lily. Your cheles can't find food and treats and read together." Lily loved the soup
```

**step 3,000**

```
Once there was a little girl named Sarah. She wanted to help her mom and help her open the cupboard toe and play! Sarah was excited and waited!
As she went into the big green cardboard box with grayly orange, it started to leaakes. Sarah was so excited that she gave it a patch of grass in
```

**step 4,000**

```
Once there was a little girl named Mary. Mary had a yellow pony. The pony was flying in a big fence. Mary loved to play with her ball and have fun. 
One day, Mary's friend Lue went for a walk and asked Lue if she wanted to get her ball. Lu sit
```

**step 5,000**

```
Once there was a little girl named Sarah and her mother went on a comfortable journey. But of the way, they saw a big surprise Sarah's house. She picked up a big egg and started to open it. Her mother helped her unpack the seas and the batter was rotting. She was amazed! 
Sarah
```

**step 6,000**

```
Once there was a little girl named Jane. She was only three years old.
One night, she asked her mom for something. She was fighting by bey and breinked. Her mom said that Jill wanted to come nap with such the sun.
Jane laughed and was very happy. But then another day she wanted
```

**step 7,000**

```
Once there was a little girl named Daisy. Daisy liked to play outside and pretending. One day, Daisy's mom put the big box to checkly. Daisy told her mom, "When we see that the box was too long and it will refuse."
She had an idea. " But I can guess who will
```

**step 8,000**

```
Once there was a little girl named Sally. Sally had a design really messy. She wanted to make something for her family, but she refused.
So, Sally decided to make something delicious. She put it in the laundry mustard and tried to part his dad's make something delicious. Billy was so pleased and proud that Sally
```

**step 9,000**

```
Once there was a little girl called Lucy. She was only three years old. She loved to wander around but one day when she was in the park, she noticed something strange. It was a box with a few thick marks colors.
Lucy was amazed when she touched the box, she tasted a bit scared. She screamed
```

**step 10,000**

```
Once there was a little girl called Mia. She was very independent. She loved her mother very much. Suzie, Lizzie’s mommy, and they agreed with her.
Mia and her mom went on a mission and sending the plan to the party to different party. Mia was very excited. In
```

Logs: `training.csv`, `per_step.csv`, `samples.txt` in `logs/` · checkpoints: `sr_data200mb_s1339_best.pt` + periodic in `sr_data200mb_s1339/` (both gitignored)
