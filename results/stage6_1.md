# Stage 4 — transformer results

## transformer_6_1 — 2026-09-21 22:31

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 10,000 steps this run (through step 10,000), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `mps`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 1.7 epochs
- wall clock: 36.1 min (217 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **2.0755** | **2.0875** | **8.1** |
| best val | — | 2.0868 | 8.1 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.0875** | **8.1** |

### Gradient norms (pre-clip, all steps)

- median 0.722, max 45.742 (63.3x median), clipped on 0.4% of steps
- **1 step(s) above 10.0x the median** — step 17 (45.7). This is the pattern the lr=3e-3 sweep run showed, where clipping hid it from the loss.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.7895 | 4.0492 | 57.3 |
| 500 | 3.8333 | 3.6062 | 36.8 |
| 750 | 3.4191 | 3.2372 | 25.5 |
| 1,000 | 3.1195 | 2.9904 | 19.9 |
| 1,250 | 2.9064 | 2.8233 | 16.8 |
| 1,500 | 2.7611 | 2.6944 | 14.8 |
| 1,750 | 2.6566 | 2.6036 | 13.5 |
| 2,000 | 2.5722 | 2.5441 | 12.7 |
| 2,250 | 2.5144 | 2.4884 | 12.0 |
| 2,500 | 2.4637 | 2.4405 | 11.5 |
| 2,750 | 2.4224 | 2.4043 | 11.1 |
| 3,000 | 2.3902 | 2.3757 | 10.8 |
| 3,250 | 2.3544 | 2.3490 | 10.5 |
| 3,500 | 2.3345 | 2.3302 | 10.3 |
| 3,750 | 2.3058 | 2.3015 | 10.0 |
| 4,000 | 2.2843 | 2.2854 | 9.8 |
| 4,250 | 2.2756 | 2.2694 | 9.7 |
| 4,500 | 2.2555 | 2.2513 | 9.5 |
| 4,750 | 2.2428 | 2.2347 | 9.3 |
| 5,000 | 2.2287 | 2.2237 | 9.2 |
| 5,250 | 2.2195 | 2.2216 | 9.2 |
| 5,500 | 2.2050 | 2.2063 | 9.1 |
| 5,750 | 2.2011 | 2.1974 | 9.0 |
| 6,000 | 2.1829 | 2.1823 | 8.9 |
| 6,250 | 2.1723 | 2.1760 | 8.8 |
| 6,500 | 2.1667 | 2.1645 | 8.7 |
| 6,750 | 2.1587 | 2.1566 | 8.6 |
| 7,000 | 2.1511 | 2.1480 | 8.6 |
| 7,250 | 2.1373 | 2.1443 | 8.5 |
| 7,500 | 2.1390 | 2.1367 | 8.5 |
| 7,750 | 2.1331 | 2.1323 | 8.4 |
| 8,000 | 2.1223 | 2.1295 | 8.4 |
| 8,250 | 2.1141 | 2.1176 | 8.3 |
| 8,500 | 2.1051 | 2.1126 | 8.3 |
| 8,750 | 2.1016 | 2.1098 | 8.2 |
| 9,000 | 2.0958 | 2.1030 | 8.2 |
| 9,250 | 2.0902 | 2.0991 | 8.2 |
| 9,500 | 2.0891 | 2.0907 | 8.1 |
| 9,750 | 2.0797 | 2.0868 | 8.1 |
| 10,000 | 2.0755 | 2.0875 | 8.1 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 1,000 -> 2,000 | 2.5441 | -0.4463 | 12.73 |
| 2,000 -> 3,000 | 2.3757 | -0.1684 | 10.76 |
| 3,000 -> 4,000 | 2.2854 | -0.0902 | 9.83 |
| 4,000 -> 5,000 | 2.2237 | -0.0617 | 9.24 |
| 5,000 -> 6,000 | 2.1823 | -0.0414 | 8.87 |
| 6,000 -> 7,000 | 2.1480 | -0.0343 | 8.57 |
| 7,000 -> 8,000 | 2.1295 | -0.0185 | 8.41 |
| 8,000 -> 9,000 | 2.1030 | -0.0265 | 8.19 |
| 9,000 -> 10,000 | 2.0875 | -0.0155 | 8.06 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Lily. She loved to play with her toys and her ball. One day, Lily's mommy said that they found her window if she couldn't find. They packed the decide to show things in the sun. They put it out at the swings for a new boy who lived with a new friend
```

**step 2,000**

```
Once there was a little girl named it. She was very excited, because she had an idea. She was so excited and ran to the garden. She used her plan to take itch and bite. She picked some honey to her mom, and she cut it with well.
At the river, she found a shiny th
```

**step 3,000**

```
Once there was a little girl who loved Lily's favorite books because it was very beautiful and she loved to spend the rest. She would throw the piano while the her mom came outside and asked her why she had to come to it. Lily felt sad because she hugged her mom and not want to.
Later that day, Lily
```

**step 4,000**

```
Once there was a little girl named Lily. She loved playing outside with her favorite soft dinosaur. One day, she asked her mommy, "Can I go to the park with your friends?". Her mommy said, "Wow, that's so cool!" 
As they walked, Lucy saw some birds dancing and they went back to
```

**step 5,000**

```
Once there was a little girl called Lucy. She was so happy that she climbed up into the sky and found her prepenter: "I love to shine up there loud people here to fove it and other kids," 
Lucy hugged her mum and set off on some mistakes. She imagined that everyone was sits
```

**step 6,000**

```
Once there was a little girl named Sarah. She was sad because the earth was fun. She was a bit scared because she wanted to stay away from it. 
One day, Sarah decided to take a bath. She found a towel and put it on their warm energy. Then she carefully looked at her parents and said
```

**step 7,000**

```
Once there was a little girl called Jane.
Jane told her veil was a very lovely drawer. She loved it very much and it made her else comfortable.
She asked her mum, "What is it?"
Her mum replied, "It's a lot of things." Jane said, "Wow, it looks perfect.
```

**step 8,000**

```
Once there was a little girl who was very sad. She asked her mom and she had a lot of fun playing with snow and snowmen and jewels and balloons.
The girl said, "Yes, Mommy! That snowmen!"
Her mom smiled and said, "I'm glad you like it, Molly
```

**step 9,000**

```
Once there was a little girl named Lily. One day, Lily woke up early because her mommy went to the valley because she had to pass. 
"Mommy, what birdie is under a door," Lily said.
"Well, sweetie, it's time to go to the valley. We have to
```

**step 10,000**

```
Once there was a little girl named Ella. She was going to the park with her mum and dad. Ella saw the sunrise and she was so happy. She felt happy and she wanted to see the park. 
When she got home, she was feeling very tired and decided to go too close to the sky.
```

Logs: `transformer_6_1_training.csv`, `transformer_6_1_grad_norms.csv`, `transformer_6_1_samples.txt` in `logs/` · checkpoints: `transformer_6_1_best.pt` + periodic in `checkpoints/` (both gitignored)
