# Stage 5 — greedy generation comparison

_2026-09-21 20:11_ · read-only evaluation, no training.

- checkpoints: **5k** (`checkpoints/transformer_best.pt`, step 5,000) vs **9.75k** (`checkpoints/transformer_10k_best.pt`, step 9,750)
- 8 fixed prompts from `eval/prompts.json`, identical for both checkpoints
- greedy decoding (argmax, no sampling), cap 150 new tokens
- deterministic: rerunning this script reproduces every generation exactly

## Metrics, averaged across all prompts

| metric | 5k | 9.75k |
|---|---|---|
| 4-gram repetition rate | 46.9% | 29.7% |
| generation length (tokens) | 111.5 | 82.4 |
| stopped at `<EOS>` | 5/8 | 5/8 |
| name tracking | recurs 2, disappears 1, no name 5 | recurs 2, disappears 1, no name 5 |
| quotation marks balanced | 6/8 | 7/8 |
| ends on terminal punctuation | 5/8 | 5/8 |

### How each metric is computed

- **4-gram repetition rate**: of all 4-grams in the generation (BPE tokens), the share of occurrences whose 4-gram appears more than once. 0% = nothing repeats.
- **Generation length**: tokens produced before the model chose `<EOS>`; equal to the 150-token cap when it never did.
- **Name tracking**: the first proper noun in prompt+generation (capitalised word, excluding sentence starters, pronouns and roles like "Mom"). *recurs* = it appears again in the generation, *changes* = it does not but another name does, *disappears* = no name after the first mention.
- **Quotation balance**: straight quotes even and curly quotes paired.
- **Ending**: last character is `.`/`!`/`?`, optionally followed by a closing quote.

## Per-prompt metrics

| prompt | ckpt | tokens | `<EOS>` | rep. | name | quotes | clean end |
|---|---|---|---|---|---|---|---|
| dog_hungry | 5k | 107 | yes | 78.8% | — | ok | yes |
| dog_hungry | 9.75k | 27 | yes | 8.3% | — | ok | yes |
| girl_store | 5k | 60 | yes | 0.0% | — | ok | yes |
| girl_store | 9.75k | 55 | yes | 0.0% | — | ok | yes |
| tom_ball | 5k | 150 | no | 85.0% | Tom (recurs) | ok | no |
| tom_ball | 9.75k | 150 | no | 79.6% | Tom (recurs) | ok | no |
| lily_key | 5k | 150 | no | 70.1% | Lily (disappears) | UNBALANCED (9) | no |
| lily_key | 9.75k | 150 | no | 82.3% | Lily (recurs) | UNBALANCED (11) | no |
| rainy_friends | 5k | 37 | yes | 0.0% | — | ok | yes |
| rainy_friends | 9.75k | 28 | yes | 0.0% | — | ok | yes |
| cat_window | 5k | 93 | yes | 47.8% | — | ok | yes |
| cat_window | 9.75k | 57 | yes | 0.0% | — | ok | yes |
| park_dialogue | 5k | 150 | no | 50.3% | Ben (recurs) | UNBALANCED (3) | no |
| park_dialogue | 9.75k | 42 | yes | 25.6% | Ben (disappears) | ok | yes |
| robot_wish | 5k | 145 | yes | 43.0% | — | ok | yes |
| robot_wish | 9.75k | 150 | no | 41.5% | — | ok | no |

## Full generations

### 1. `dog_hungry` — animal subject, causal continuation

**Prompt:** `The dog was hungry, so it`

**5k** (107 tokens, stopped at <EOS>, repetition 78.8%):

```
The dog was hungry, so it barked loudly.
The dog barked loudly and ran away. The dog was scared and ran away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog chased the dog away. The dog caught the dog away. The dog was safe. The dog was happy.
```

**9.75k** (27 tokens, stopped at <EOS>, repetition 8.3%):

```
The dog was hungry, so it barked and ran away. The dog was so happy that he had saved the dog. The dog was happy that he helped the dog.
```

### 2. `girl_store` — unnamed character, motivation clause

**Prompt:** `The girl went to the store because`

**5k** (60 tokens, stopped at <EOS>, repetition 0.0%):

```
The girl went to the store because she was very kind. She asked her mom if she could go. Her mom said yes and they went to the store.
When they arrived, the store was full of toys and toys. The girl was so happy and thanked her mom for the help. She thanked her mom and went home.
```

**9.75k** (55 tokens, stopped at <EOS>, repetition 0.0%):

```
The girl went to the store because she was very happy. She had a new toy and a new toy. She was so happy and hugged her mom.
The girl was so happy that she had her toy. She hugged her mom and said, "Thank you, mom. I love my new toy."
```

### 3. `tom_ball` — named character introduced in the prompt, action continuation

**Prompt:** `Tom picked up the red ball and`

**5k** (150 tokens, hit the length cap, repetition 85.0%):

```
Tom picked up the red ball and threw it at the ball. He threw it at the ball and ran to the ball. He ran to the ball and threw it at the ball. He ran to the ball and threw it at the ball. He ran to the ball and tried to catch it.
Tom and Lily were scared. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did
```

**9.75k** (150 tokens, hit the length cap, repetition 79.6%):

```
Tom picked up the red ball and threw it to the ball. He threw it to the ball and threw it back. He threw the ball to the ball and threw it back. He threw the ball back and threw it back.
Tom and Lily were very sad and angry. They wanted to play with the ball. They did not want to lose the ball. They did not want to lose the ball. They did not want to lose the ball. They did not want to lose it. They did not want to lose the ball. They did not want to lose the ball. They did not want to lose it. They did not want to lose the ball. They did not want to lose it. They did not want to lose the ball. They did not want to lose
```

### 4. `lily_key` — named character across a sentence boundary, pronoun reference

**Prompt:** `Lily found a shiny key under the bed. She`

**5k** (150 tokens, hit the length cap, repetition 70.1%):

```
Lily found a shiny key under the bed. She thought it was a good idea. She picked it up and put it in her pocket. She put it in her pocket and went to the park. She saw a big tree with a sign that said "Do not open it. It is a big tree. It has a sign that says "Do not open it. It is a sign that says "Do not open it. It is a sign that says "Do not open. It is a sign that says "Do not open. It is not open. It is a sign that says "Do not open. It is a sign that says "Do not open. It is a sign that says "Do not open. It is not open. It is a sign that says "Do not
```

**9.75k** (150 tokens, hit the length cap, repetition 82.3%):

```
Lily found a shiny key under the bed. She was very happy. She hugged her mom and said, "I love you, mom. I love you."
Her mom smiled and said, "I love you too, Lily. You are a very good girl. You are a good girl. You are a very good girl."
Lily smiled and said, "I love you, mom. I love you. I love you."
Her mom said, "I love you too, Lily. You are a good girl. You are a good girl. I love you."
Lily said, "I love you too, mom. I love you. I love you."
Her mom said, "I love you too, Lily. You are a good girl. You are a good
```

### 5. `rainy_friends` — plural subject, indoor scene

**Prompt:** `It was raining outside, so the two friends stayed inside and`

**5k** (37 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and the rain was safe.
The little girl was so happy that she had to go outside and play with her friends. She was so happy and proud of herself for being so careful.
```

**9.75k** (28 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and watched the rain.
The rain was so happy that it was safe and they stayed inside for a while, and the rain was safe.
```

### 6. `cat_window` — scene setting followed by a turn word

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

**5k** (93 tokens, stopped at <EOS>, repetition 47.8%):

```
The old cat sat by the window watching the birds. Suddenly, the cat heard a loud noise. It was coming from a big tree. The cat jumped out of the window and flew away.
The cat was very scared. It was a big, scary dog. The dog ran away. The dog chased the cat away. The dog chased the cat away. The dog chased the cat away. The dog chased the cat away. The dog chased the cat away. The dog chased the cat away. The dog was happy.
```

**9.75k** (57 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the cat heard a loud noise. It was a big dog. The cat was barking and looked scared. The cat ran away and the cat ran away.
The cat was happy to be safe. The cat and the cat were safe and happy. They had a new friend.
```

### 7. `park_dialogue` — quoted speech already closed in the prompt, then narration

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

**5k** (150 tokens, hit the length cap, repetition 50.3%):

```
Mom said, "We are going to the park today!" Ben was so happy that he said, "I'm glad you had to go home."
They went to the park and saw a big tree. It was a big tree with a big tree. It was very big and shiny. Ben and Lily were curious. They wanted to see what was inside.
They saw a man with a hat and a hat. The man was holding a hat and a hat. He said, "Hello, little girl. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park. I am the park
```

**9.75k** (42 tokens, stopped at <EOS>, repetition 25.6%):

```
Mom said, "We are going to the park today!" Ben was so happy that he had a new friend. He said, "I'm glad you found a new friend!"
They went to the park and played together. They had a lot of fun. They had a lot of fun.
```

### 8. `robot_wish` — classic story opening, non-animal character, desire premise

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

**5k** (145 tokens, stopped at <EOS>, repetition 43.0%):

```
Once upon a time, there was a tiny robot who wanted to go on a journey. He was so excited to find out what was going on the way.
He asked his friend, "What is that?"
The robot replied, "I can help you find it."
The robot replied, "I can help you find it."
The robot was so happy and said, "Thank you, little robot. You can help me find it."
The robot was so happy and thanked the robot. He thanked the robot and said, "You're welcome. I'm glad you had helped me find your way home."
The robot smiled and said, "Thank you for helping me find your way home."
The robot was happy and thanked the robot for his help.
```

**9.75k** (150 tokens, hit the length cap, repetition 41.5%):

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He asked his friends if they could go on an adventure.
His friends said yes, but they were too little to go on an adventure. They went on a journey and found a big, shiny object.
The little robot was so excited to explore the world. He wanted to explore the world, so he started to explore.
He found a big, shiny object and he was so excited. He wanted to explore it. He carefully picked it up and started to explore.
He found a big, shiny object and he was so excited. He couldn't wait to explore the world around him.
He was so excited to explore the world around him. He had never seen anything so shiny before. He was
```

