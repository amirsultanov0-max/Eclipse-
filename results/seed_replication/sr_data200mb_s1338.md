# Seed replication — 200 MB (6.1), seed 1338

## sr_data200mb_s1338 — 2026-09-22 20:02

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 10,000 steps this run (through step 10,000), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1338, device `mps`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 1.7 epochs
- wall clock: 39.3 min (236 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **2.0807** | **2.0919** | **8.1** |
| best val | — | 2.0911 | 8.1 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.0919** | **8.1** |

### Gradient norms (pre-clip, all steps)

- median 0.710, max 8.604 (12.1x median), clipped on 0.3% of steps
- **2 step(s) above 10.0x the median** — step 25 (8.6), step 30 (8.6). This is the pattern the lr=3e-3 sweep run showed, where clipping hid it from the loss.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.7891 | 4.0513 | 57.5 |
| 500 | 3.8377 | 3.5958 | 36.4 |
| 750 | 3.4011 | 3.2105 | 24.8 |
| 1,000 | 3.0918 | 2.9748 | 19.6 |
| 1,250 | 2.8935 | 2.8232 | 16.8 |
| 1,500 | 2.7514 | 2.6932 | 14.8 |
| 1,750 | 2.6553 | 2.6010 | 13.5 |
| 2,000 | 2.5697 | 2.5391 | 12.7 |
| 2,250 | 2.5110 | 2.4849 | 12.0 |
| 2,500 | 2.4601 | 2.4397 | 11.5 |
| 2,750 | 2.4197 | 2.4080 | 11.1 |
| 3,000 | 2.3918 | 2.3716 | 10.7 |
| 3,250 | 2.3549 | 2.3489 | 10.5 |
| 3,500 | 2.3250 | 2.3208 | 10.2 |
| 3,750 | 2.3106 | 2.2994 | 10.0 |
| 4,000 | 2.2873 | 2.2849 | 9.8 |
| 4,250 | 2.2746 | 2.2679 | 9.7 |
| 4,500 | 2.2569 | 2.2486 | 9.5 |
| 4,750 | 2.2493 | 2.2413 | 9.4 |
| 5,000 | 2.2304 | 2.2293 | 9.3 |
| 5,250 | 2.2130 | 2.2173 | 9.2 |
| 5,500 | 2.2093 | 2.2052 | 9.1 |
| 5,750 | 2.2007 | 2.1964 | 9.0 |
| 6,000 | 2.1958 | 2.1892 | 8.9 |
| 6,250 | 2.1762 | 2.1754 | 8.8 |
| 6,500 | 2.1640 | 2.1669 | 8.7 |
| 6,750 | 2.1557 | 2.1651 | 8.7 |
| 7,000 | 2.1525 | 2.1507 | 8.6 |
| 7,250 | 2.1427 | 2.1452 | 8.5 |
| 7,500 | 2.1390 | 2.1409 | 8.5 |
| 7,750 | 2.1360 | 2.1391 | 8.5 |
| 8,000 | 2.1228 | 2.1291 | 8.4 |
| 8,250 | 2.1203 | 2.1203 | 8.3 |
| 8,500 | 2.1119 | 2.1186 | 8.3 |
| 8,750 | 2.1134 | 2.1135 | 8.3 |
| 9,000 | 2.1046 | 2.1060 | 8.2 |
| 9,250 | 2.0941 | 2.1062 | 8.2 |
| 9,500 | 2.0902 | 2.1002 | 8.2 |
| 9,750 | 2.0862 | 2.0911 | 8.1 |
| 10,000 | 2.0807 | 2.0919 | 8.1 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 1,000 -> 2,000 | 2.5391 | -0.4356 | 12.67 |
| 2,000 -> 3,000 | 2.3716 | -0.1675 | 10.71 |
| 3,000 -> 4,000 | 2.2849 | -0.0868 | 9.82 |
| 4,000 -> 5,000 | 2.2293 | -0.0555 | 9.29 |
| 5,000 -> 6,000 | 2.1892 | -0.0402 | 8.93 |
| 6,000 -> 7,000 | 2.1507 | -0.0385 | 8.59 |
| 7,000 -> 8,000 | 2.1291 | -0.0216 | 8.41 |
| 8,000 -> 9,000 | 2.1060 | -0.0230 | 8.22 |
| 9,000 -> 10,000 | 2.0919 | -0.0141 | 8.10 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl. The girl was little with her owner. One day, a big and a loyal person asked the boy to carry the ant. The music was so excited! 
The girl thought it would make her to put the earth pizza splash and make bright strawberry cre. She was so excited to go
```

**step 2,000**

```
Once there was a little girl named Samie was very nervous. He shouted to a big plane as he tried to pair, but he lost his hand. 
"Sorry, camp a brilliant concane," Samrisy and the plane continued together.” 
But Sam in a calm voice: "You brothers
```

**step 3,000**

```
Once there was a little girl who loved to explore. Every day she would go explore, do space-se it up, exploring the world was filled by nature.
One day her parents wanted a finding the in the country every day. She decided nobody to solve the generous stories, so they could feel the tree eas
```

**step 4,000**

```
Once there was a little girl named Sperer.
Every day, time she went, she would go outside and explore the world. One day, her mum was looking around and around for Rex. They were very scared, but miged that Rex couldn't move back. 
Lila was so excited to discover new
```

**step 5,000**

```
Once there was a little girl. She was three years old and she liked to play very special in the living room. One day, she met a little bird. The little birdie wanted to take it home. The bird said, “Stop fighting, stop fighting the little birdie. The birdie said to the birdie should take the mole
```

**step 6,000**

```
Once there was a little girl who was three years old. Every day she would pick a new ornaments in her pocket. Mommy had a long green ornament that she couldn't put it together forever. The little girl was very happy because it felt amazing! 
Mommy loved taking little things by putting small things in the pretty ornament
```

**step 7,000**

```
Once there was a little girl called Sally. Sally loved wearing fancy suit and shoes. One day, Sally saw a big, red truck driving by. Sally wanted to have a bus please so she asked her mommy for help. Sally was so excited! 
After the bus, Sally sat down in the roof and ate a lot
```

**step 8,000**

```
Once there was a little girl called Daisy. She loved to spin around and spin around and go around. One day, Daisy saw something red in the sky. She got some and jumped even harder.
The little girl chirped and ran over. But she couldn't resist her mom. Then, she heard a loud noise
```

**step 9,000**

```
Once there was a little girl named Lily. She was frustrated because she couldn't do anything fit there. 
One day, Lily's mom asked her to help her help with help. Lily sang a great song and sang a happy song. 
After that, Lily's mom bought her a new new story. Lily showed
```

**step 10,000**

```
Once there was a little girl named Maria. She lived in the park with her daughter, Tom. Jane loved to play and drink juice with all of her cereal. One day, while playing, she needed some milk, Tom went to visit her grandma. It saw that there was a yummy milk with a special bowl of milk.
```

Logs: `training.csv`, `per_step.csv`, `samples.txt` in `logs/` · checkpoints: `sr_data200mb_s1338_best.pt` + periodic in `sr_data200mb_s1338/` (both gitignored)
