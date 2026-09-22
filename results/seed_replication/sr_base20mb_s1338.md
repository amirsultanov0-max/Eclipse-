# Seed replication — 20 MB baseline, seed 1338

## sr_base20mb_s1338 — 2026-09-22 19:19

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 10,000 steps this run (through step 10,000), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1338, device `mps`
- data: `data/stage3_train_tokens.npy` — 4,691,267 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 17.5 epochs
- wall clock: 36.8 min (221 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **1.9775** | **2.2293** | **9.3** |
| best val | — | 2.2293 | 9.3 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.2293** | **9.3** |

### Gradient norms (pre-clip, all steps)

- median 0.734, max 25.112 (34.2x median), clipped on 0.3% of steps
- **1 step(s) above 10.0x the median** — step 26 (25.1). This is the pattern the lr=3e-3 sweep run showed, where clipping hid it from the loss.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.7872 | 4.0632 | 58.2 |
| 500 | 3.8427 | 3.6293 | 37.7 |
| 750 | 3.4099 | 3.2756 | 26.5 |
| 1,000 | 3.0982 | 3.0219 | 20.5 |
| 1,250 | 2.8894 | 2.8621 | 17.5 |
| 1,500 | 2.7515 | 2.7564 | 15.7 |
| 1,750 | 2.6439 | 2.6688 | 14.4 |
| 2,000 | 2.5666 | 2.6196 | 13.7 |
| 2,250 | 2.4972 | 2.5684 | 13.0 |
| 2,500 | 2.4450 | 2.5292 | 12.5 |
| 2,750 | 2.3999 | 2.4949 | 12.1 |
| 3,000 | 2.3663 | 2.4673 | 11.8 |
| 3,250 | 2.3271 | 2.4480 | 11.6 |
| 3,500 | 2.2972 | 2.4243 | 11.3 |
| 3,750 | 2.2755 | 2.3996 | 11.0 |
| 4,000 | 2.2485 | 2.3880 | 10.9 |
| 4,250 | 2.2216 | 2.3678 | 10.7 |
| 4,500 | 2.2076 | 2.3647 | 10.6 |
| 4,750 | 2.1910 | 2.3444 | 10.4 |
| 5,000 | 2.1769 | 2.3336 | 10.3 |
| 5,250 | 2.1637 | 2.3270 | 10.2 |
| 5,500 | 2.1362 | 2.3124 | 10.1 |
| 5,750 | 2.1328 | 2.3058 | 10.0 |
| 6,000 | 2.1143 | 2.3018 | 10.0 |
| 6,250 | 2.1045 | 2.2905 | 9.9 |
| 6,500 | 2.0895 | 2.2869 | 9.8 |
| 6,750 | 2.0867 | 2.2774 | 9.8 |
| 7,000 | 2.0734 | 2.2699 | 9.7 |
| 7,250 | 2.0641 | 2.2670 | 9.7 |
| 7,500 | 2.0493 | 2.2581 | 9.6 |
| 7,750 | 2.0397 | 2.2613 | 9.6 |
| 8,000 | 2.0324 | 2.2494 | 9.5 |
| 8,250 | 2.0247 | 2.2471 | 9.5 |
| 8,500 | 2.0125 | 2.2443 | 9.4 |
| 8,750 | 2.0150 | 2.2373 | 9.4 |
| 9,000 | 2.0004 | 2.2391 | 9.4 |
| 9,250 | 1.9979 | 2.2316 | 9.3 |
| 9,500 | 1.9946 | 2.2350 | 9.3 |
| 9,750 | 1.9852 | 2.2319 | 9.3 |
| 10,000 | 1.9775 | 2.2293 | 9.3 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 1,000 -> 2,000 | 2.6196 | -0.4024 | 13.73 |
| 2,000 -> 3,000 | 2.4673 | -0.1523 | 11.79 |
| 3,000 -> 4,000 | 2.3880 | -0.0794 | 10.89 |
| 4,000 -> 5,000 | 2.3336 | -0.0544 | 10.31 |
| 5,000 -> 6,000 | 2.3018 | -0.0318 | 9.99 |
| 6,000 -> 7,000 | 2.2699 | -0.0318 | 9.68 |
| 7,000 -> 8,000 | 2.2494 | -0.0205 | 9.48 |
| 8,000 -> 9,000 | 2.2391 | -0.0103 | 9.39 |
| 9,000 -> 10,000 | 2.2293 | -0.0099 | 9.29 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl. The girl was so happy. 
The mommy said thanks and the girl loved going on the street and became his wings to music and they lived so many friends.
They played saw red medal toys and played together. It was bright and different cre soar in their marks. The
```

**step 2,000**

```
Once there was a little girl named Benie called very nervous. He had to be careful with me. It was a lot, but he lost her hand and closed his eyes. His eyes lit up and the stage washingriverse into her mouth. 
Judy opened in a corner and saw that it was
```

**step 3,000**

```
Once there was a little girl named Lucy. She replied to Jane, "Let's always do something yummy."
One day, Lucy and her mom baked a big bowl of grapes. Lucy started eating the tasty sandwich. They took out some grapes and soon eating grapes. On the camera, they tried to feed the tree all
```

**step 4,000**

```
Once there was a little girl named Bella who loved picking the stuff and she loved her new stuff. One day, her neighbor's owner noticed something unusual in the backyard was missing from the cart. Rex barked and Lily liked it, but she was nowhere to get lost.
A lady saw a boy named Lily and tried to pet
```

**step 5,000**

```
Once there was a little girl. Her name was Lisa and she was feeling a very comfort of welcomers. 
One night, Lisa heard a big storm coming. She opened the door and looked inside. She looked around her room. 
Jake opened his door to look, but she found a strange room full
```

**step 6,000**

```
Once there was a little girl named Mary. Mary was so excited to get her waffle, because she had bought some lollipops for breakfast too. The next day her friends came and together came to teach her by taking her cookie. Mary felt happy and excited to have her waffle.
After small drink, Mary showed her
```

**step 7,000**

```
Once there was a little girl called Amy. Amy loved ever, and knew what to do. One day, she found one of her when she wanted. 
"Mommy, what's in the kitchen when you look for her?" said Amy. 
Mommy said, "Don't give up, sweetie. Said it
```

**step 8,000**

```
Once there was a little girl called Lucy. She loved to spin around and strang behind the curtains. One night, Lucy had to go to bed early! She had to fly even higher and higher.
When she finally got outside, her living room was so big! Lucy was so excited that she started to spin!
```

**step 9,000**

```
Once there was a little girl named Garia. She was three years old and very loved to explore nice things. One day, she went outside to explore.
Suddenly, Garia went to an island. She saw a weird statue on the earth that made her mom happy. Garia wanted to examine a
```

**step 10,000**

```
Once there was a little girl named Anna. Lily loved to play by herself. One day, her little brother came to them to play with all of her toys. Anna was so excited when her brother gave her a shot!
"Anna, please, where your toyientist?" her brother asked.
"Is
```

Logs: `training.csv`, `per_step.csv`, `samples.txt` in `logs/` · checkpoints: `sr_base20mb_s1338_best.pt` + periodic in `sr_base20mb_s1338/` (both gitignored)
