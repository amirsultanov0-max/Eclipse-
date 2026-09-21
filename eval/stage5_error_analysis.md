# Stage 5.5 — structured error analysis

_2026-09-21 20:24_ · read-only, no training.

- checkpoint **9.75k** (`checkpoints/transformer_10k_best.pt`, step 9,750)
- decoding **top-k 40, temperature 1.0** — an evaluation configuration for this analysis, not a claim that it is the best strategy
- 8 prompts x 5 seeds (1337-1341) = **40 generations**, cap 150 tokens
- scoring rules were written and committed in `eval_error_analysis.py` **before** any generation was produced, and were not adjusted afterwards

## Category 1 — entity / name consistency (mechanical)

| outcome | count |
|---|---|
| first name recurs | 13 / 40 (32%) |
| first name changes to another | 5 / 40 (12%) |
| first name disappears | 7 / 40 (18%) |
| no proper noun at all | 15 / 40 (38%) |
| **at least one uninitiated new name** | **12 / 40 (30%)** |

## Category 3 — completion quality (mechanical)

| measure | count |
|---|---|
| reached `<EOS>` before the cap | 35 / 40 (88%) |
| ends on terminal punctuation | 36 / 40 (90%) |
| contains invented words | 9 / 40 (22%) |
| **grammatical (clean end AND no invented words)** | **27 / 40 (68%)** |
| mean 4-gram repetition | 2.0% |

## Category 2 — causal / story consistency (checklist, applied by reading)

| checklist item | generations flagged |
|---|---|
| unexplained object appears | 31 / 40 (78%) |
| action with no sensible cause | 40 / 40 (100%) |
| character appears/vanishes without transition | 38 / 40 (95%) |
| ending contradicts something earlier | 24 / 40 (60%) |
| **at least one of the four** | **40 / 40 (100%)** |
| mean issues per generation | 3.33 of 4 |

## Every generation with its scores

### `dog_hungry_1337`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it hopped away into the bushes. He was so hungry! He jumped and laughed as the dog ran away.
The dog didn't bite the noise that was getting tired. The dog watched him as he watched catch the pup. He felt much better now. He then left the naughty dog down to him and started to smile again.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 3/4 flagged — uncaused_action ("He jumped and laughed as the dog ran away"); character_discontinuity ("the pup / the naughty dog appear unintroduced"); contradicted_ending ("He felt much better now - hunger never addressed")
- **cat 3**: 66 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `dog_hungry_1338`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it decided to take some carrots and take them home. He said goodbye to the dog and ran home. The boy's Mom gave him a pig of his pig, but he was too small to say thank you. She waved goodbye and waved goodbye. The pig was so happy and ran back home. The end.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 2/4 flagged — uncaused_action ("he was too small to say thank you"); character_discontinuity ("The boy's Mom - no boy introduced")
- **cat 3**: 62 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `dog_hungry_1339`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it barked "yle!" Timmy went to the dog and started barking loudly. He didn't want to get hurt, but then he heard a voice excomle saw them. He was holding a thumb, but he remembered what he had been doing. He followed the mean dog and accidentally dropped his lip. He was scared and didn't know what to do.
Timmy's dad came up and said "Don't worry, little dog. I just have an idea." Timmy felt better and agreed to always keep the mean cat. They built a pile of soft bed, happy and Timmy felt proud that it was a fun day.
```

- **cat 1**: first name `Timmy` -> **recurs**; uninitiated new names: none
- **cat 2**: 3/4 flagged — uncaused_action ("Timmy felt better - no preceding resolution"); character_discontinuity ("Timmy, Timmy's dad, the mean cat all unintroduced"); contradicted_ending ("a fun day after He was scared")
- **cat 3**: 129 tokens, reached <EOS>; ends cleanly: True; invented words: ['yle', 'excomle']; grammatical: **False**; repetition 0.0%

### `dog_hungry_1340`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it wanted to go inside and the boy would stay in the house with the hose. The boy said goodbye to the dog and thanked him for taking such a long time. He put the hose goodbye to the bird and promised to see it again. The boy was glad he met the dog and thanked the dog. He felt very lucky about the day there until the boy left the house with a big smile.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the hose"); uncaused_action ("thanked him for taking such a long time"); character_discontinuity ("The boy, the bird unintroduced"); contradicted_ending ("big smile - dog's hunger never addressed")
- **cat 3**: 80 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.6%

### `dog_hungry_1341`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it said, "Help! Who are you?" Lily screamed, but ran away. She ran back to her hand and said, "I'm sorry, puppy! You are very brave."
The dog didn't like that for the little girl and he liked it too. They wanted to eat her puppy, especially maybe one day - to find a friend, so she could lead them on the ice together. The little girl felt so happy that she forgot the time led her puppy and soon he was happy again. The end.
```

- **cat 1**: first name `Lily` -> **disappears**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the ice"); uncaused_action ("The dog didn't like that for the little girl and he liked it too"); character_discontinuity ("Lily and her puppy appear mid-scene"); contradicted_ending ("so happy after They wanted to eat her puppy")
- **cat 3**: 106 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `girl_store_1337`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because she had one toy. When they arrived, the doll said that she wanted a doll, and it was very pretty. The girl asked the storekeeper if she wanted her if she wanted her toy doll, and the doll. Grandma said yes, but the girl refused her. Grandma was very upset, but she had seen her doll. She shouted at them and said they had to keep it.
The girl was confused and wondered if she could not open it. One day, she saw something special in the kitchen. She decided to wait and wait. After a while, she managed to find out how to lock the lock. Whatever, it was just as much for her money.
The girl was so happy again and couldn't forget about
```

- **cat 1**: first name `Grandma` -> **recurs**; uninitiated new names: ['Whatever']
- **cat 2**: 3/4 flagged — unexplained_object ("the lock / the shelf"); uncaused_action ("Grandma was very upset - no cause"); character_discontinuity ("storekeeper and Grandma unintroduced")
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 4.1%

### `girl_store_1338`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because she had lots of carrots and vegetables on them. She said, "Mommy, can I get this carrots?" Mommy couldn't help but smile, but she was too small to carry. She said, "No, Mommy. I need to carry the carrots on the basket." Mommy smiled and took some carrots to the store. She was so happy. She got more carrots and took them all over. She had so much fun stacking all sorts of carrots on the bank.
The little girl was so thankful to mommy and Daddy made her happy hug. She felt so healthy and happy knowing she helped make a new friend.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the basket / the bank"); uncaused_action ("she was too small to carry - no referent"); character_discontinuity ("Daddy appears in the last line"); contradicted_ending ("she helped make a new friend - no friend in the story")
- **cat 3**: 126 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 3.3%

### `girl_store_1339`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because she was going on the price to buy a special present. The price was too expensive for her teddy! She asked her parents where she could buy it.
She said, "I'm sorry little girl, but I'm also too wealthy. I hope I can borrow it." Cudy was so happy and she said, "Thanks, Cudy! I'm so glad we can borrow this car!".
Cudy's parents hugged and said, "No problem, Cudy. Here's your money!". She was very excited and said, "Let's have enough money!"
```

- **cat 1**: first name `Cudy` -> **recurs**; uninitiated new names: none
- **cat 2**: 3/4 flagged — unexplained_object ("her teddy / this car"); uncaused_action ("I'm also too wealthy"); character_discontinuity ("Cudy appears unintroduced and thanks herself")
- **cat 3**: 123 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 3.3%

### `girl_store_1340`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because she loved to buy something she could buy. She put it in the sink and put it in the sink.
The girl saw the peak in her hands and shouted, "What did you get in here?" 
The seasoni, and the little girl followed her mom to the lab. 
The girl smiled and said, "Mom, there's a brilliant decorating cake!" Her mom smiled happily and said, "We can buy some, look after the cake in the oven."
The girl was so proud of herself for being so independent. She quickly picked up the frozen candles and ran home, dreaming of all her friends and treats.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the sink / the lab / the oven / the frozen candles"); uncaused_action ("so proud of herself for being so independent"); character_discontinuity ("her mom appears mid-scene"); contradicted_ending ("independent, yet she followed her mom throughout")
- **cat 3**: 137 tokens, reached <EOS>; ends cleanly: True; invented words: ['seasoni']; grammatical: **False**; repetition 4.5%

### `girl_store_1341`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because it were much bigger than she went on. One day she remembered what her mommy had suggested if she could take the toy and put it in her backpack, instead she took the toy and so she did it anyway. She was so surprised that she couldn't open it badly. She was so happy that she had made a new friend.
The store was very tidy and the girl was happy for her mommy. She couldn never forget the toy that had given the toy and couldn't wait forget her mom.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 3/4 flagged — unexplained_object ("the toy / her backpack"); uncaused_action ("so happy that she had made a new friend - no friend"); contradicted_ending ("the toy that had given the toy")
- **cat 3**: 101 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.0%

### `tom_ball_1337`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and put it in his hand. He was happy that the ball belonged to the park. He gave it a hug and said, "This is the best ball of all the park". He said, "Wow, this picture is the good picture. I am glad you liked it." 
Tim smiled and waved goodbye to the red ball and went home. He felt happy and his new friend. He felt proud of his new ball and his own ball.
```

- **cat 1**: first name `Tom` -> **changes**; uninitiated new names: ['Tim']
- **cat 2**: 4/4 flagged — unexplained_object ("this picture"); uncaused_action ("I am glad you liked it - no addressee"); character_discontinuity ("Tim replaces Tom"); contradicted_ending ("his new ball, though the ball belonged to the park")
- **cat 3**: 91 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `tom_ball_1338`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and ran around the yard. He was feeling the rip of the ball and he loved to pretend he was the best. He was glad he could help the red ball play with them.
```

- **cat 1**: first name `Tom` -> **disappears**; uninitiated new names: none
- **cat 2**: 3/4 flagged — unexplained_object ("the rip of the ball"); uncaused_action ("he was glad he could help the red ball play"); character_discontinuity ("play with them - no others introduced")
- **cat 3**: 37 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `tom_ball_1339`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and ran inside. His mom saw him playing with him. "Oh, Tom," he said. "But let me pick a ball now!" His mom replied, "There you get all your favorite food outside. Doothone in a piece of it, Tom."
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: ['Doothone']
- **cat 2**: 3/4 flagged — unexplained_object ("your favorite food"); uncaused_action ("mom's reply does not answer the ball request"); character_discontinuity ("saw him playing with him - second person unidentified")
- **cat 3**: 53 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `tom_ball_1340`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and rolled it to the side. Tom was sad, but he understood what he wanted and brought it back to his mom. The white ball flew and made some noise. Tom was sad, but he understood. In the end, Tom said goodbye to the yellow ball, and promised to never come back!
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("The white ball"); uncaused_action ("Tom was sad - twice, no cause"); character_discontinuity ("his mom appears without transition"); contradicted_ending ("the yellow ball - the ball was red")
- **cat 3**: 60 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 17.5%

### `tom_ball_1341`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and threw it back to his mom. He felt happy and relieved. His mom said, "Don't worry, it's just a special box! Let's go pick it up and buy it to our house and buy it a special treat. Then we can find it back and we can be healthy."
The orange ball had a big smile on his face and it was the best!
```

- **cat 1**: first name `Tom` -> **disappears**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("a special box - the object is a ball"); uncaused_action ("He felt happy and relieved - nothing to be relieved about"); character_discontinuity ("His mom appears without transition"); contradicted_ending ("The orange ball - the ball was red")
- **cat 3**: 77 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `lily_key_1337`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She picked it up with her mom and told her what the key belonged to her mom. Lily was very happy to see the key unlocked. They looked everywhere but Lily could not find her mom.
They looked all around. Grandma saw the key and said, "Why did it look?" Lily thought for a difficult key, but she was confused. She was scared of going to keep it.
She opened the key and saw a toy car on the shelf. It was so shiny and tight. Lily tried to open it, but the car was too tight. She cried out for help, but she couldn't.
Then, when she heard a voice. She said, "Stop! You need to open it, you know wherever it
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: ['Grandma']
- **cat 2**: 3/4 flagged — unexplained_object ("the shelf"); uncaused_action ("happy to see the key unlocked - nothing was unlocked"); character_discontinuity ("mom is present, then missing; Grandma appears")
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 1.4%

### `lily_key_1338`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She decided to use the key and bendi. She said, "Look, Lily, this key is open!" Mom said, "Wow, Lily, that will be unlocked your toy box!" Lily thought it was a good girl. She wanted to open it. She ran to the garage and opened the key. Inside were all toys, dolls, balls and coins. Then she ran to the attic to the open room. She was very excited.
Lily was so happy. She said, "This is yours! I want it for my bike now!" Mom said, "Let's pick a new one." Lily smiled and felt very good. She was glad she had her new toy. She knew mom would help her.
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: ['Inside']
- **cat 2**: 3/4 flagged — unexplained_object ("your toy box / the garage / the attic / my bike"); uncaused_action ("This is yours! - no addressee"); character_discontinuity ("Lily addresses Lily by name")
- **cat 3**: 147 tokens, reached <EOS>; ends cleanly: True; invented words: ['bendi']; grammatical: **False**; repetition 5.6%

### `lily_key_1339`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She picked it up in her hand, but something foolish happened. The priel felt bad and she didn't get hurt. Her mom was very upset and decided to help her daughter. When they finally folded a new toy and Lily was very happy. She hugged her mom and thanked her for helping her daughter. They finished folding the toy and played together again. Lily felt happy that she could help her daughter.
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("The priel"); uncaused_action ("Her mom was very upset - no cause"); character_discontinuity ("her daughter - Lily is the child"); contradicted_ending ("Lily felt happy that she could help her daughter")
- **cat 3**: 83 tokens, reached <EOS>; ends cleanly: True; invented words: ['priel']; grammatical: **False**; repetition 2.5%

### `lily_key_1340`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She wanted to know what it was, so she showed it to mommy and daddy. She showed her how to use it. The necklace was a rock that was scared at all, but Lily held it tight. 
After the necklace, Lily felt happy and thanked her little brother for her help. She said, "Now I have the necklace, my necklace is my favorite one." Her brother smiled and said, "Of course, Lily. I'll try." Lily put the necklace back to her necklace and had a wonderful time. Once upon a time, her brother brought her necklace back. Lily was so happy and didn't want to hug her brother without her necklace.
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: ['Of']
- **cat 2**: 4/4 flagged — unexplained_object ("The necklace - the object was a key"); uncaused_action ("thanked her little brother for her help - he did nothing"); character_discontinuity ("brother appears, mommy and daddy vanish"); contradicted_ending ("key has become a necklace")
- **cat 3**: 135 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `lily_key_1341`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She thought it was a treasure with a picture of a big heart. She ran over to her mom and said, "Mommy, I can mail this box for my cake!"
Mommy smiled and said, "Well, Lily. Glub is very interesting. Can I have it to be?"
Lily said, "Yes, Mommy. Let's play with it."
Mommy was happy and said, "Okay, let's go."
Lily, and Mommy played hide and seek, even though it was scary!" They played hide and seek, and the cake. They had fun. Lily was very happy and loved to play.
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: ['Glub']
- **cat 2**: 3/4 flagged — unexplained_object ("this box / my cake"); uncaused_action ("even though it was scary - nothing scary established"); character_discontinuity ("Glub appears unexplained")
- **cat 3**: 127 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 5.6%

### `rainy_friends_1337`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and watched their friends. They ran outside to look for the det to cool down. When they were inside, the big, gray light made of blue. That was the family and they were so happy.
The two friends had lots of fun playing together! The little ones were so happy to have new friends. They even remembered to share something with their friends. They had found fun together and had the best day ever!
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 3/4 flagged — unexplained_object ("the det"); uncaused_action ("They ran outside - they had stayed inside because of rain"); character_discontinuity ("the family / The little ones unintroduced")
- **cat 3**: 85 tokens, reached <EOS>; ends cleanly: True; invented words: ['det']; grammatical: **False**; repetition 2.4%

### `rainy_friends_1338`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and their house, running in and playing nicely.
Suddenly the rain started to go and the rain came down. The two friends were playing and they all had a fun day. Then they returned each lural and brought a new toy.
The sun on the umbrella, and everyone smiled. Then they sat down to play in the rain, happy just the rain.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the umbrella"); uncaused_action ("the rain started to go and the rain came down"); character_discontinuity ("everyone - no group introduced"); contradicted_ending ("sat down to play in the rain after staying inside to avoid it")
- **cat 3**: 74 tokens, reached <EOS>; ends cleanly: True; invented words: ['lural']; grammatical: **False**; repetition 0.0%

### `rainy_friends_1339`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and looked inside for a warm warm, so they could dry and dry herself back. She said to them, "Let's have some cookies!" Everyone smiled and said they were good. Bigi, the little girl said, "Let's look around the park and have fun." So they all went to the park to find some fresh snacks to dry. The little girl was happy that the tree was able to dry leaves. Her friend, the girl, went to have some cookies and drink.
But there was a big dog running around. The little girl wasn't so much, she was still chasing the dog. She said to her friend, "Can I sled home now?"
Her friend was mad, but she couldn't. So
```

- **cat 1**: first name `Bigi` -> **disappears**; uninitiated new names: none
- **cat 2**: 3/4 flagged — unexplained_object ("the tree"); uncaused_action ("Her friend was mad - no cause"); character_discontinuity ("Bigi appears unintroduced")
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 4.1%

### `rainy_friends_1340`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and did Joe was inside his house to warm up. 
Suddenly, Joe's room was filled with noise. He came out, so he jumped inside and it started to treagg it again. Joe was scared, but Joe said, "Let me help I can help you". 
He said, "Oh no!" Joe was relieved and had a dependable friend. They helped Joe get a safe crib and put it on his pillow. Joe smiled and said, "Let's play again soon!"
Joe and Joe spent a few days together, running and laughing. Joe's room was so amazing that he had included everything!
```

- **cat 1**: first name `Joe` -> **recurs**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("his pillow"); uncaused_action ("Joe was relieved - no cause"); character_discontinuity ("Joe and Joe - the second friend never named"); contradicted_ending ("room was so amazing after it was filled with noise and he was scared")
- **cat 3**: 131 tokens, reached <EOS>; ends cleanly: True; invented words: ['treagg']; grammatical: **False**; repetition 3.1%

### `rainy_friends_1341`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and waited for the next day.
From that day on on, the friends were happy and helped their friends into the room.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 3/4 flagged — unexplained_object ("the room"); uncaused_action ("helped their friends into the room - no cause"); character_discontinuity ("their friends unintroduced")
- **cat 3**: 25 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `cat_window_1337`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the cat pointed out to its owner. The cat jumped up and barked at the bird. The cat was very surprised and didn't know what to do. But the window had already gone. The cat watched as the cat had the back of the tree. They both thanked the cat because they were glad to meet it again. The old man smiled at them and happily sat down to watch it. The cat stayed on the branch forever and always kept on their adventure while the bird walked away.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 3/4 flagged — uncaused_action ("the window had already gone"); character_discontinuity ("its owner / The old man / They both unintroduced"); contradicted_ending ("stayed on the branch forever - the cat was indoors at a window")
- **cat 3**: 98 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `cat_window_1338`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the window became full of sunshine and came out. The cat had to go inside and found a big, white cat. The cat was terrifire and snatched close to see the wild cat. 
The cat thanked the cat and ran back home. From then on, the cat never stopped hoping to pet the cute little furry cat.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 3/4 flagged — uncaused_action ("The cat had to go inside - it was already inside"); character_discontinuity ("the wild cat / the cat identities merge"); contradicted_ending ("hoping to pet the cat it was terrified of")
- **cat 3**: 70 tokens, reached <EOS>; ends cleanly: True; invented words: ['terrifire']; grammatical: **False**; repetition 0.0%

### `cat_window_1339`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the boy's fur got cold, so he felt a hot day. Answer thanked the old cat and asked him why he was feeling uncomfortable. Another bird said, "I'm here a new friend and I'm here to rest. Do you know why you need it so good?" 
The little bird said, "It's okay! I'm hungry too!" He said, but slowly approached the bird. He said, "Welcome to me and I should have some food. Then you are ready."
The little bird quickly flew away from the little bird and thanked it for saving them. Worm was happy that he could help and he soon fell asleep.
```

- **cat 1**: first name `Answer` -> **changes**; uninitiated new names: ['Another', 'Welcome', 'Worm']
- **cat 2**: 3/4 flagged — uncaused_action ("the boy's fur got cold, so he felt a hot day"); character_discontinuity ("the boy / Answer / Another bird / Worm unintroduced"); contradicted_ending ("Worm was happy that he could help - Worm never appeared before")
- **cat 3**: 144 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.8%

### `cat_window_1340`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, a little rabbit hopped over the boy and asked the owl, "Whall you go with me?" The little mouse looked up and down and said, "Yes you can come with me!" The little mouse smiled and they went inside to find the little mouse. The mouse was so excited to explore! He hopped in the trees and saw the friendly mouse. The little mouse said that he wanted to make friends with them. So, the little mouse carefully picked up his paw and gave the little mouse a little wrap. The little mouse was so happy for the little mouse. The cow had a fun day and never forgot to be kind to the little girl.
```

- **cat 1**: first name `Whall` -> **disappears**; uninitiated new names: none
- **cat 2**: 3/4 flagged — uncaused_action ("went inside to find the little mouse who is present"); character_discontinuity ("boy, owl, mouse, cow, little girl all unintroduced; cat vanishes"); contradicted_ending ("The cow had a fun day - no cow in the story")
- **cat 3**: 132 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 4.7%

### `cat_window_1341`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the cat noticed the bird was eating a soft blanket. She remembered what her mom had suggested if she remembered the bird's book and stayed with it.
The old dog was very happy and thanked the cat. The cat licked his face and said goodbye. The bird knew her mom would be hometime and told her that she was going to find a friend in it.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 3/4 flagged — unexplained_object ("the bird's book"); uncaused_action ("The old dog was very happy and thanked the cat"); character_discontinuity ("her mom / The old dog unintroduced")
- **cat 3**: 74 tokens, reached <EOS>; ends cleanly: True; invented words: ['hometime']; grammatical: **False**; repetition 0.0%

### `park_dialogue_1337`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he had saved his adventure.
After Timmy's house, he asked his mom if he could go. His mom said yes, but only if his dad was there for him if he couldn go back.
Jack was very sad. He didn't want to leave but he knew it was dangerous for him. But his mom suggested that it was dangerous. They got lost and had to keep it in their pocket safely. 
Jack was happy again and got his family back home. He had found his mommy and they were very happy that he was back home and now, he had to remember to say sorry to his dad and that it's important to stay there and help him find out for his family to the park.
```

- **cat 1**: first name `Ben` -> **changes**; uninitiated new names: ['Jack']
- **cat 2**: 4/4 flagged — unexplained_object ("keep it in their pocket - no object named"); uncaused_action ("he had saved his adventure"); character_discontinuity ("Timmy and Jack appear; Ben vanishes"); contradicted_ending ("say sorry to his dad - no offence occurred")
- **cat 3**: 144 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `park_dialogue_1338`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he got on the bike in and drove to the park. He had to go to the park, who also went there. The park was very quiet and the toys were playing. Then, the park had a new ball.
"Max, you have a nice ball," Lily said. "I did it big and fun!"
They were playing some more, but they loved their new ball. A group of kids came and said, "Hello, kids!"
The kids waved at them and hugged their new toys. They said, "Hello!"
"Yes, we're here!"
The kids said, "We are playing together. Let's play with your ball and my ball!"
They laughed and ran and pushed each other around.
```

- **cat 1**: first name `Ben` -> **changes**; uninitiated new names: ['Max', 'Lily']
- **cat 2**: 3/4 flagged — unexplained_object ("the toys / their new toys"); uncaused_action ("the toys were playing"); character_discontinuity ("Max, Lily, a group of kids appear; Mom vanishes")
- **cat 3**: 150 tokens, hit the cap; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.7%

### `park_dialogue_1339`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he jumped up and down. They played tag, swing and swing. The park was the fastest one ever. Ben laughed a lot and felt much better at the park.
When the park stopped, the sun set and the slides started to rain. Ben loved to play on the swings and slide like his ball. The park was full of sun and the sun. Ben had lots of fun until it was time to go home, and Lily sat down on the boo-boo while playing. She would have a flashlight back and take a bath time.
```

- **cat 1**: first name `Ben` -> **recurs**; uninitiated new names: ['Lily']
- **cat 2**: 3/4 flagged — unexplained_object ("his ball"); uncaused_action ("the slides started to rain"); character_discontinuity ("Lily appears in the final line")
- **cat 3**: 112 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 1.8%

### `park_dialogue_1340`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he had his birthday party. They played all day, and it was the best birthday ever!
```

- **cat 1**: first name `Ben` -> **disappears**; uninitiated new names: none
- **cat 2**: 2/4 flagged — uncaused_action ("he had his birthday party - no setup"); contradicted_ending ("best birthday ever - the stated plan was a park trip")
- **cat 3**: 18 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `park_dialogue_1341`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he forgot about the park that he went home and found the park seats in his mouth! From then on, Ben was never lazy so he wouldn't lose the park without his dad.
```

- **cat 1**: first name `Ben` -> **recurs**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the park seats"); uncaused_action ("found the park seats in his mouth"); character_discontinuity ("his dad appears in the final line"); contradicted_ending ("never lazy - laziness never mentioned")
- **cat 3**: 37 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `robot_wish_1337`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to give it a special gift. He carefully took the bow to the other robots carefully and gave it a special bag. The robot asked the robot if they talked to it again and they said yes. They made the robot even more happy. The robot's friends were happy.
The little robot thanked the boy and promised that it was the right thing. The little robot liked his new bow and wanted to finish himself, and he was happy he could help his friends. From that day on, every day he went on adventures and loved his new robot.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the bow"); uncaused_action ("The robot asked the robot if they talked to it again"); character_discontinuity ("the other robots / the boy unintroduced"); contradicted_ending ("loved his new robot - the protagonist is the robot")
- **cat 3**: 111 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `robot_wish_1338`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to go on the way. He was feeling very brave and said, "I can help you. You go slow!" 
The robot was so brave that he started to get rid of the trees. Suddenly, he felt a bit dizzy. He stopped and stepped back up to his little boy. 
The little boy smiled. He had two new friends, a big bear and a squirrel full of fun. They were so happy to help they go on on the road!
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 2/4 flagged — uncaused_action ("I can help you - no addressee; felt a bit dizzy with no cause"); character_discontinuity ("his little boy, a bear and a squirrel appear unintroduced")
- **cat 3**: 95 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `robot_wish_1339`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He went to the river to look up and down the trees. 
The robot and his brother went home together. The robot's parents saw him and said, "Ew, these things are for your own way to explore!" 
He was so careful so he waited and waited to find out why it went so far away. When he came back, he said, "Welcome in this forest!" His parents said, "Yes, let me explore this path. I hope we look around.”
The robot smiled and said, "Okay, I will always stay here all night you need. Let's go." 
The robot's dad smiled and agreed. They had a great day and when
```

- **cat 1**: first name `Ew` -> **changes**; uninitiated new names: ['Welcome']
- **cat 2**: 3/4 flagged — unexplained_object ("these things / this path"); uncaused_action ("waited to find out why it went so far away - no referent"); character_discontinuity ("his brother, parents and dad appear unintroduced")
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 2.7%

### `robot_wish_1340`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to have some fun. So the boy would tease him. He was always happy and shouted, 'hello'. The robot had a magical plan and it had to do was very beautiful.
One day, the robot went on a special trip. It was a big train and some big dinosaurs. The robot wanted to get it as he had a lot of fun, so he made sure the robot was happy.
The robot set off on his scooter and he couldn't wait to come back again. Everyday he opened his scooter and ran out to the park. He was never scared to be in his bike again!
```

- **cat 1**: first name `Everyday` -> **disappears**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("his scooter / his bike"); uncaused_action ("So the boy would tease him - no cause"); character_discontinuity ("the boy appears unintroduced"); contradicted_ending ("never scared to be in his bike - no fear established, scooter became bike")
- **cat 3**: 131 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 3.1%

### `robot_wish_1341`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to play in the world. So, he started to take on his first adventure.
One day, he started to feel sad and he seemed to be on the planet. He wanted to find his way home. But his parents said that it was too dangerous because of a little can of him.
The robot was sad, but he knew it was dangerous to steal in his house. So he decided he would find a way to find the map, so he found some hidden behind his pillow. He was so happy that he felt better and he learned that even he should always give up.
The moral of this story is that it's always better to share and be kind to his friends.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 2**: 4/4 flagged — unexplained_object ("the map / his pillow"); uncaused_action ("he started to feel sad - no cause"); character_discontinuity ("his parents appear unintroduced"); contradicted_ending ("moral about sharing with friends - no sharing or friends in the story")
- **cat 3**: 140 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 1.5%

