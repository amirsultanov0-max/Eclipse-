# Seed replication — 200 MB (6.1), seed 1337

## sr_data200mb_s1337 — 2026-09-22 18:39

- model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head — 1,767,424 parameters
- config: 10,000 steps this run (through step 10,000), batch 16, lr 0.001, AdamW (weight decay 0.01 on projection/FFN weights only), grad clip 1.0, seed 1337, device `mps`
- data: `data/stage6_1_train_tokens.npy` — 49,124,477 train tokens; validation is the Stage 3 monitoring split (`tinystories_valid.txt` untouched)
- saw 81,920,000 tokens = 1.7 epochs
- wall clock: 116.5 min (699 ms/step)

| loss (nats) | train | val | val perplexity |
|---|---|---|---|
| **final (step 10,000)** | **2.0845** | **2.0860** | **8.1** |
| best val | — | 2.0860 | 8.1 |

### Reference points

Not a fair head-to-head: the bigram sees 1 token of context, this model sees up to 512. The comparison is architectural.

| model | val loss | val perplexity |
|---|---|---|
| uniform, log(4000) | 8.2940 | 4000 |
| unigram (Stage 3) | 6.0253 | 413.8 |
| trained bigram (Stage 3, best run) | 4.6354 | 103.1 |
| count-based bigram floor (Stage 3) | 3.7715 | 43.4 |
| **this transformer** | **2.0860** | **8.1** |

### Gradient norms (pre-clip, all steps)

- median 0.722, max 45.742 (63.4x median), clipped on 0.4% of steps
- **1 step(s) above 10.0x the median** — step 17 (45.7). This is the pattern the lr=3e-3 sweep run showed, where clipping hid it from the loss.

### Loss curve

| step | train | val | val ppl |
|---|---|---|---|
| 250 | 4.7895 | 4.0492 | 57.3 |
| 500 | 3.8333 | 3.6062 | 36.8 |
| 750 | 3.4192 | 3.2373 | 25.5 |
| 1,000 | 3.1195 | 2.9904 | 19.9 |
| 1,250 | 2.9125 | 2.8232 | 16.8 |
| 1,500 | 2.7672 | 2.7008 | 14.9 |
| 1,750 | 2.6539 | 2.6124 | 13.6 |
| 2,000 | 2.5780 | 2.5379 | 12.7 |
| 2,250 | 2.5176 | 2.4857 | 12.0 |
| 2,500 | 2.4581 | 2.4394 | 11.5 |
| 2,750 | 2.4235 | 2.4060 | 11.1 |
| 3,000 | 2.3896 | 2.3656 | 10.7 |
| 3,250 | 2.3521 | 2.3450 | 10.4 |
| 3,500 | 2.3303 | 2.3186 | 10.2 |
| 3,750 | 2.3100 | 2.3045 | 10.0 |
| 4,000 | 2.2923 | 2.2868 | 9.8 |
| 4,250 | 2.2760 | 2.2732 | 9.7 |
| 4,500 | 2.2647 | 2.2507 | 9.5 |
| 4,750 | 2.2411 | 2.2372 | 9.4 |
| 5,000 | 2.2271 | 2.2249 | 9.3 |
| 5,250 | 2.2084 | 2.2202 | 9.2 |
| 5,500 | 2.2080 | 2.2065 | 9.1 |
| 5,750 | 2.1948 | 2.1934 | 9.0 |
| 6,000 | 2.1901 | 2.1793 | 8.8 |
| 6,250 | 2.1709 | 2.1749 | 8.8 |
| 6,500 | 2.1628 | 2.1696 | 8.8 |
| 6,750 | 2.1495 | 2.1564 | 8.6 |
| 7,000 | 2.1421 | 2.1525 | 8.6 |
| 7,250 | 2.1421 | 2.1430 | 8.5 |
| 7,500 | 2.1304 | 2.1373 | 8.5 |
| 7,750 | 2.1273 | 2.1331 | 8.4 |
| 8,000 | 2.1192 | 2.1264 | 8.4 |
| 8,250 | 2.1136 | 2.1184 | 8.3 |
| 8,500 | 2.1075 | 2.1186 | 8.3 |
| 8,750 | 2.1002 | 2.1054 | 8.2 |
| 9,000 | 2.0941 | 2.1014 | 8.2 |
| 9,250 | 2.0927 | 2.0978 | 8.1 |
| 9,500 | 2.0851 | 2.0960 | 8.1 |
| 9,750 | 2.0921 | 2.0934 | 8.1 |
| 10,000 | 2.0845 | 2.0860 | 8.1 |

### Validation improvement per 1,000 steps

| segment | val loss | improvement | ppl |
|---|---|---|---|
| 1,000 -> 2,000 | 2.5379 | -0.4525 | 12.65 |
| 2,000 -> 3,000 | 2.3656 | -0.1724 | 10.65 |
| 3,000 -> 4,000 | 2.2868 | -0.0788 | 9.84 |
| 4,000 -> 5,000 | 2.2249 | -0.0618 | 9.25 |
| 5,000 -> 6,000 | 2.1793 | -0.0456 | 8.84 |
| 6,000 -> 7,000 | 2.1525 | -0.0268 | 8.61 |
| 7,000 -> 8,000 | 2.1264 | -0.0261 | 8.38 |
| 8,000 -> 9,000 | 2.1014 | -0.0251 | 8.18 |
| 9,000 -> 10,000 | 2.0860 | -0.0154 | 8.05 |

### Samples from "Once there was a little girl" (temperature 1.0, qualitative only)

**step 1,000**

```
Once there was a little girl named Lucy. She was three years old and a little girl named Bill. One day, Jack heard a voice calling him. She saw a round bow mintion on its face and tried to admire it. 
The little girl felt bad and clapped on her crumated. She felt
```

**step 2,000**

```
Once there was a little girl named Lily. She was a very kindest, because of pretty things that it was a thoughtful girl. 
One day, Lily went to the park with her mommy. When she saw the girl running to keep looking for her her. Her mommy said, "Pat Catch You must
```

**step 3,000**

```
Once there was a little girl named Anna. She was very excited, and she was looking at the new places. Later she saw a big caterpillar peeshing its name. Lily was happy and asked if she could do it. The caterpillar replied that she didn't mind to mix the caterpillar legs.
The caterpillar hopped towards
```

**step 4,000**

```
Once there was a little girl named Lily. She loved to play outside in her puddles and so she would bounce pastries. Suddenly, she heard a noise coming from inside. It was coming closer and Lily didn't know what a reily thing was. She realized her mommy had coming closer and picked her up it up,
```

**step 5,000**

```
Once there was a little girl named Lily. Lily loved her picture of the pictures in the jar and the flowers. One day, she saw a bird that had yellow a picture of a jar of a banana. Lily carefully picked it up and noticed it was a bee that sparkled in its feathers. They were safe and nice.

```

**step 6,000**

```
Once there was a little girl named Mandy who was very hairy. One day she decided to explore the world, around her house to see what was going to happen. Her mom warned her to be careful and not too slow she decided to learn.
The little girl was very curious and she asked her mom what it was.
```

**step 7,000**

```
Once there was a little girl named Ella. Ella went to the market and found an vegetables. Ella had a knife and dug from the cupboard.
Ella had an idea: she said: Emily: to her mum: “Mom, look! Please mix something up here, cut down?" 
Her Dad was amb
```

**step 8,000**

```
Once there was a little girl named Lola. She wanted some cream up to her mom, because it was blue. Her mom would order the shed and put it under her mommy's mouth. Lola was very enthusiastic and didn't have any friends anymore. But one day, all the toys needed to eat together. Lola
```

**step 9,000**

```
Once there was a little girl who was very curious. She wanted to have a fun catch, but every morning she found her friend Lucy. She tried to explain to her friend, but it twisted her strength and scratched her body. Lucy was sad because she wanted to give her friend a prize. And every
```

**step 10,000**

```
Once there was a little girl who loved to eat mining things. She would part for dressing them and look for mint. One day, she was as bright as she was hanging out of lemon, she would stain her abilitities made one of the lith quixture.
She
```

Logs: `training.csv`, `per_step.csv`, `samples.txt` in `logs/` · checkpoints: `sr_data200mb_s1337_best.pt` + periodic in `sr_data200mb_s1337/` (both gitignored)
