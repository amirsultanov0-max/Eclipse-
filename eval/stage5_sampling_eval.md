# Stage 5b — sampling strategies vs greedy

_2026-09-21 20:16_ · read-only evaluation, no training.

- prompts: the frozen set in `eval/prompts.json` (same 8 as the greedy baseline)
- cap 150 new tokens, identical for every strategy
- **seeds: `torch.manual_seed(1337 + sample_index)`** before each generation, so samples 1-3 always use 1337, 1338, 1339 — the same seeds across every strategy and prompt, and reruns reproduce exactly
- 3 samples per (prompt, strategy); greedy is deterministic so it has one
- scope: **9.75k** on all 8 prompts; **5k** on the two prompts that looped worst under greedy (tom_ball, lily_key)

Repetition = share of 4-gram occurrences in the generation whose 4-gram appears more than once, the same metric as the greedy baseline.

## 9.75k across all 8 prompts — mean 4-gram repetition

| prompt | greedy | temp 0.7 | temp 1.0 | top-k 40 | top-p 0.9 |
|---|---|---|---|---|---|
| dog_hungry | 8.3% | 0.0% | 1.3% | 0.0% | 2.2% |
| girl_store | 0.0% | 13.9% | 4.4% | 3.6% | 4.2% |
| tom_ball | 79.6% | 11.1% | 0.5% | 0.0% | 10.4% |
| lily_key | 82.3% | 2.1% | 2.2% | 3.1% | 1.4% |
| rainy_friends | 0.0% | 1.5% | 0.0% | 2.2% | 0.0% |
| cat_window | 0.0% | 7.3% | 0.0% | 0.9% | 0.0% |
| park_dialogue | 25.6% | 1.8% | 1.5% | 1.5% | 0.6% |
| robot_wish | 41.5% | 8.2% | 3.1% | 0.9% | 2.9% |
| **mean** | **29.7%** | **5.7%** | **1.6%** | **1.5%** | **2.7%** |
| mean length | 82 | 94 | 111 | 108 | 102 |
| stopped at `<EOS>` | 5/8 | 19/24 | 18/24 | 19/24 | 18/24 |

## The two looping prompts, per sample

Each cell is one generation's repetition rate, at the seed shown.

| prompt | ckpt | strategy | seed 1337 | seed 1338 | seed 1339 | mean |
|---|---|---|---|---|---|---|
| tom_ball | 5k | greedy | 85.0% | — | — | 85.0% |
| tom_ball | 5k | temp 0.7 | 15.0% | 5.3% | 16.3% | 12.2% |
| tom_ball | 5k | temp 1.0 | 5.4% | 0.0% | 0.0% | 1.8% |
| tom_ball | 5k | top-k 40 | 9.5% | 0.0% | 0.0% | 3.2% |
| tom_ball | 5k | top-p 0.9 | 5.4% | 0.0% | 6.8% | 4.1% |
| tom_ball | 9.75k | greedy | 79.6% | — | — | 79.6% |
| tom_ball | 9.75k | temp 0.7 | 8.8% | 15.9% | 8.5% | 11.1% |
| tom_ball | 9.75k | temp 1.0 | 1.5% | 0.0% | 0.0% | 0.5% |
| tom_ball | 9.75k | top-k 40 | 0.0% | 0.0% | 0.0% | 0.0% |
| tom_ball | 9.75k | top-p 0.9 | 14.3% | 12.9% | 4.1% | 10.4% |
| lily_key | 5k | greedy | 70.1% | — | — | 70.1% |
| lily_key | 5k | temp 0.7 | 0.0% | 1.4% | 6.8% | 2.7% |
| lily_key | 5k | temp 1.0 | 0.0% | 5.9% | 2.7% | 2.9% |
| lily_key | 5k | top-k 40 | 0.0% | 6.8% | 0.0% | 2.3% |
| lily_key | 5k | top-p 0.9 | 0.0% | 1.4% | 5.6% | 2.3% |
| lily_key | 9.75k | greedy | 82.3% | — | — | 82.3% |
| lily_key | 9.75k | temp 0.7 | 3.4% | 0.0% | 3.0% | 2.1% |
| lily_key | 9.75k | temp 1.0 | 3.2% | 3.3% | 0.0% | 2.2% |
| lily_key | 9.75k | top-k 40 | 1.4% | 5.6% | 2.5% | 3.1% |
| lily_key | 9.75k | top-p 0.9 | 1.5% | 2.7% | 0.0% | 1.4% |

## Full generations

## Checkpoint 9.75k (`checkpoints/transformer_10k_best.pt`, step 9,750)

### `dog_hungry` — prompt: `The dog was hungry, so it`

**greedy**, sample 1 (deterministic, 27 tokens, stopped at <EOS>, repetition 8.3%):

```
The dog was hungry, so it barked and ran away. The dog was so happy that he had saved the dog. The dog was happy that he helped the dog.
```

**temp 0.7**, sample 1 (seed 1337, 51 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it hopped away. The dog ran away, but the dog was too quick.
The dog and the dog became friends. They shared the dog together every day. The dog had a lot of fun. They ate delicious food and the boy learned that day.
```

**temp 0.7**, sample 2 (seed 1338, 53 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it decided to take a nap and begging for the dog. He was happy to eat and eat the nap. The dog ate the nap and they both cheered for their friendship. From that day on, they were known as friends and were the best of friends.
```

**temp 0.7**, sample 3 (seed 1339, 45 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it barked "Welcome to the dog". 
The dog wagged its tail and they ate their food and went back to the park. The dog was still hungry, but he was happy to have a new friend.
```

**temp 1.0**, sample 1 (seed 1337, 55 tokens, stopped at <EOS>, repetition 3.8%):

```
The dog was hungry, so it hopped away into the rough oxygen. He said hello to the dog, and gave him a big bag. The dog barked back, but then appeared. The dog barked all the way away. He was very sad that he didn't like the wild dog anymore.
```

**temp 1.0**, sample 2 (seed 1338, 93 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it decided to take some carrots and be friendly. During the provide the bell and shouted, "Oh no! The house belongs so noisy! I will show you our carrots nice and nice stories!".
The dog wagged his tail and jumped, on the leash, laughing with joy. After the ride, the driver drove to his two new friends' yard. They had never seen an A group of old kitten when the boy came out of the playground.
```

**temp 1.0**, sample 3 (seed 1339, 68 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it barked "yle!" Timmy went to the swing and started to run back. He played with his toys and played with him. The dog showed Timmy how to ride the dog. Timmy was happy too because he was enough to ride a piece of the truck. From that day on, Timmy always rode his motorcycle and even threw his ball.
```

**top-k 40**, sample 1 (seed 1337, 66 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it hopped away into the bushes. He was so hungry! He jumped and laughed as the dog ran away.
The dog didn't bite the noise that was getting tired. The dog watched him as he watched catch the pup. He felt much better now. He then left the naughty dog down to him and started to smile again.
```

**top-k 40**, sample 2 (seed 1338, 62 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it decided to take some carrots and take them home. He said goodbye to the dog and ran home. The boy's Mom gave him a pig of his pig, but he was too small to say thank you. She waved goodbye and waved goodbye. The pig was so happy and ran back home. The end.
```

**top-k 40**, sample 3 (seed 1339, 129 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it barked "yle!" Timmy went to the dog and started barking loudly. He didn't want to get hurt, but then he heard a voice excomle saw them. He was holding a thumb, but he remembered what he had been doing. He followed the mean dog and accidentally dropped his lip. He was scared and didn't know what to do.
Timmy's dad came up and said "Don't worry, little dog. I just have an idea." Timmy felt better and agreed to always keep the mean cat. They built a pile of soft bed, happy and Timmy felt proud that it was a fun day.
```

**top-p 0.9**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 2.7%):

```
The dog was hungry, so it hopped away into the pool. He was so excited to explore the pool.
The two friends followed the dog out of the pool. The two of three of them liked counting and they enjoyed the swim. Every time they had the best time.
The owner and he visited the pool, the dog and he visited the big pool. They looked at each other and smiled together. It was a fun game.
The dog had a wonderful day and he was so excited to jump out of the water. He went for a swim and watched the little duck swim close to him.
One day, the dog saw the little boy and said, "Hi there! Would you like that for some you?" He said yes, and he licked the whole
```

**top-p 0.9**, sample 2 (seed 1338, 90 tokens, stopped at <EOS>, repetition 0.0%):

```
The dog was hungry, so it decided to take some carrots and be friendly. The dog said, "I recommend this carrots. I should gather those carrots and they are very kind. What should we do?" 
The dog was surprised and wagged his tail. The dog was so happy and ran back home to tell his mom about the carrots. His mom was happy too. They all hugged and said, "See, don't spend these carrots around the yard!"
```

**top-p 0.9**, sample 3 (seed 1339, 109 tokens, stopped at <EOS>, repetition 3.8%):

```
The dog was hungry, so it barked "yle!" Timmy went to the swing and started to run back. He played with his toys and played with him. The dog loved his game and he saw a little mouse hopping in the mud. He said, "Hello, Mr. Reblp! Do you want to play with me?" Timmy said, "Yes, please!" So, Timmy played a game of hide-and-seek with the mouse in the mud. He was so excited to pretend that he was playing and having lots of fun. The end.
```

### `girl_store` — prompt: `The girl went to the store because`

**greedy**, sample 1 (deterministic, 55 tokens, stopped at <EOS>, repetition 0.0%):

```
The girl went to the store because she was very happy. She had a new toy and a new toy. She was so happy and hugged her mom.
The girl was so happy that she had her toy. She hugged her mom and said, "Thank you, mom. I love my new toy."
```

**temp 0.7**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 17.0%):

```
The girl went to the store because she had a big bag of money. She wanted to buy a new one, but she couldn't find it. She felt miserable.
Suddenly, she heard her mummy's phone and ran to see if it was her. She saw the man and said, "Why are you so sad?"
The boy replied, "I'm sorry, I was upset because it's there's it wearing a new shirt like a toy."
The boy smiled and said, "I'm sorry, I'm sorry I didn't mean to."
The boy said, "I'm glad you're sorry, but it's okay to take it when you're more careful."
The boy smiled and said, "Thank you for telling me. That's a
```

**temp 0.7**, sample 2 (seed 1338, 123 tokens, stopped at <EOS>, repetition 9.2%):

```
The girl went to the store because she had lots of money and she could buy the cake. She was so happy and ran home. She went home and introduced all of her money. She couldn't wait to buy the cake and it was a big cake. She was a little bit scared when she saw the cake.
When the cake was ready, the little girl got her cake and took it out of the cake. She was so happy and safe. She thanked the little girl for being so kind and hugged her little girl. She had a wonderful day and she hugged her sun and went inside to celebrate her delicious cake.
```

**temp 0.7**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 15.6%):

```
The girl went to the store because she was right. But she didn't know what to do. She started to worry and she didn't know what to do. She asked her mom if she could have a cookie, but she didn't know what to do. 
Then, she had an idea. She said it would be a special treat if she wanted to play with. She said it would be fun, so she said yes. She was so happy and played with her toys all day.
The girl spent the rest of the afternoon playing with the cookie, being a nice and careful not to lose it. She was thankful for the treat, then she got to play with her toys. She had so much fun playing with her new toy and couldn't wait to tell
```

**temp 1.0**, sample 1 (seed 1337, 98 tokens, stopped at <EOS>, repetition 10.5%):

```
The girl went to the store because she had one mailed it with lots of letters. She asked her mom if she could borrow it for her. 
She asked her mom if she could borrow it if she couldn't as much. Her mom said yes and gave the girl some coins! The little girl was very happy for borrowing her pocket.
The girl smiled admired her shiny gold. She had found a itch and felt very proud of herself. She had trading such a special sweet little too.
```

**temp 1.0**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 2.7%):

```
The girl went to the store because she had lots of carrots and vegetables on them. She said, "Mommy, can I get this carrots?" Mommy couldn't help but smile they would get a healthy bowl of carrots. She handed the carrots, but soon the girl was crying. She was a naughty girl, laughing with joy. 
The girl stopped calling her mommy: "Mommy, can we share this sack?".
Her mommy said, "Of course, we can be here from the market nearby. Have a tasty food to visit for someone." 
The girl was proud of herself. Together, they got to the sack. Everywhere they went, eggs and mummed the most sweet sausages wereners! When the house was full, she couldn't wait
```

**temp 1.0**, sample 3 (seed 1339, 118 tokens, stopped at <EOS>, repetition 0.0%):

```
The girl went to the store because she was right in the hurry, so she got to the store. A magical place, she saw lots of different things - a golden crown, an old spot for a long time. She called the police, but nobody had to tell it. So the store owner just said it was okay to let it to serve dinner. She told the little girl that she had saved the money world a place to make the family in the little girl's house. The little girl said hello and promised to help. With a big help, she returned the money to serve the girl.
```

**top-k 40**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 4.1%):

```
The girl went to the store because she had one toy. When they arrived, the doll said that she wanted a doll, and it was very pretty. The girl asked the storekeeper if she wanted her if she wanted her toy doll, and the doll. Grandma said yes, but the girl refused her. Grandma was very upset, but she had seen her doll. She shouted at them and said they had to keep it.
The girl was confused and wondered if she could not open it. One day, she saw something special in the kitchen. She decided to wait and wait. After a while, she managed to find out how to lock the lock. Whatever, it was just as much for her money.
The girl was so happy again and couldn't forget about
```

**top-k 40**, sample 2 (seed 1338, 126 tokens, stopped at <EOS>, repetition 3.3%):

```
The girl went to the store because she had lots of carrots and vegetables on them. She said, "Mommy, can I get this carrots?" Mommy couldn't help but smile, but she was too small to carry. She said, "No, Mommy. I need to carry the carrots on the basket." Mommy smiled and took some carrots to the store. She was so happy. She got more carrots and took them all over. She had so much fun stacking all sorts of carrots on the bank.
The little girl was so thankful to mommy and Daddy made her happy hug. She felt so healthy and happy knowing she helped make a new friend.
```

**top-k 40**, sample 3 (seed 1339, 123 tokens, stopped at <EOS>, repetition 3.3%):

```
The girl went to the store because she was going on the price to buy a special present. The price was too expensive for her teddy! She asked her parents where she could buy it.
She said, "I'm sorry little girl, but I'm also too wealthy. I hope I can borrow it." Cudy was so happy and she said, "Thanks, Cudy! I'm so glad we can borrow this car!".
Cudy's parents hugged and said, "No problem, Cudy. Here's your money!". She was very excited and said, "Let's have enough money!"
```

**top-p 0.9**, sample 1 (seed 1337, 99 tokens, stopped at <EOS>, repetition 0.0%):

```
The girl went to the store because she had one toy. When they arrived, the doll said that she wanted a doll, and it was very pretty. The girl asked the storekeeper if she wanted her doll, but the storekeeper wouldn't pay for her toy doll. The girl was sad but her wealthy doll was gone.
The boy had seen her doll and smiled with joy. The girl felt happy knowing she could share her doll with the money and it had been trading too hard to make someone else sad.
```

**top-p 0.9**, sample 2 (seed 1338, 68 tokens, stopped at <EOS>, repetition 0.0%):

```
The girl went to the store because she had lots of carrots and vegetables on them. She said, "Mommy, can I get this carrots?" Mommy said, "Yes, of one of those carrots are for our carrots." 
The girl was so happy that she started gathering all the carrots. She ran back home with them. It was the best day ever!
```

**top-p 0.9**, sample 3 (seed 1339, 114 tokens, stopped at <EOS>, repetition 12.6%):

```
The girl went to the store because she was right. But she didn't.
"What's wrong, the nurse?" she asked.
"I lost my favorite toy. Do you know where I can find it, please," the little girl said.
"Don't worry, we can try," the little girl said.
The little girl smiled and she gave the toy. The little girl was happy that the kind old girl was okay. She took the toy car and ran back home.
The little girl took the toy car back and continued the old maning her magic toy car.
```

### `tom_ball` — prompt: `Tom picked up the red ball and`

**greedy**, sample 1 (deterministic, 150 tokens, hit the cap, repetition 79.6%):

```
Tom picked up the red ball and threw it to the ball. He threw it to the ball and threw it back. He threw the ball to the ball and threw it back. He threw the ball back and threw it back.
Tom and Lily were very sad and angry. They wanted to play with the ball. They did not want to lose the ball. They did not want to lose the ball. They did not want to lose the ball. They did not want to lose it. They did not want to lose the ball. They did not want to lose the ball. They did not want to lose it. They did not want to lose the ball. They did not want to lose it. They did not want to lose the ball. They did not want to lose
```

**temp 0.7**, sample 1 (seed 1337, 116 tokens, stopped at <EOS>, repetition 8.8%):

```
Tom picked up the red ball and threw it back. He ran away with the red ball. He hid behind a tree and looked for a turn. He saw a small boy with a big smile. He had a big smile and a small smile. He had a big smile on his face. He said he was very cute.
The boy nodded and let go of Tom's hand. He said they were his friend it was his best friend. They both had a lot of fun together. Tom and Lily felt happy. They learned to be kind and help others. They learned to listen and not get naughty.
```

**temp 0.7**, sample 2 (seed 1338, 91 tokens, stopped at <EOS>, repetition 15.9%):

```
Tom picked up the red ball and ran to the ball. He was happy.
"Look, Tom, look at my ball!" Lily said. "I found it first!"
"I found it!" Tom said. "I found it first!"
They played with the red ball on the wall. They argued and argued until the ball broke. Tom's mom was happy. They thought they were playing with the ball again. They were having fun with the red ball.
```

**temp 0.7**, sample 3 (seed 1339, 74 tokens, stopped at <EOS>, repetition 8.5%):

```
Tom picked up the red ball and ran to play in the snow, but it was hard and hard to shake. He tried to grab it, but he couldn't. He tried to get away but he couldn't. 
Finally, he had enough of his ball to the snow. He was so happy that he played soccer with it. He was having so much fun. The end.
```

**temp 1.0**, sample 1 (seed 1337, 136 tokens, stopped at <EOS>, repetition 1.5%):

```
Tom picked up the red ball and put it in his hand. He was happy that the ball belonged to the park. He accepted the ball for what you can. He said he was happy that it had a lot of fun. He said the picture had the missing one. He said he would find out that he decided to remember where he had seen the last one and admired it.
He sat down his bike itch and felt himself like good things. He looked at the picture together, and when he tried to do it again. He planned a small puppy and thanked them for helping him find his favorite toy. He was glad that he had been honest and had his puppy trusted his grandma.
```

**temp 1.0**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 0.0%):

```
Tom picked up the red ball and ran around the yard. He was feeling ign as they drove for a long time and found a spot on the street. The grass was very quiet and the toys were calm. Then, the little zoom rolled away with him. The big skelet were sad, but he understood no matter wherever he went.
When the meeting the moon, something hot happened. His pants were never the same. He had to stop when the sky came out of the sky. It was a big yellow ball! Both the ball was still an enchilded now because the sun would glow too.
The little kangaroo swallowed sparkled because it was the most magical shiny pink ball he had ever seen. So, and this time, it became
```

**temp 1.0**, sample 3 (seed 1339, 56 tokens, stopped at <EOS>, repetition 0.0%):

```
Tom picked up the red ball and ran inside. His mom saw him playing with him. "Oh, Tom," he said. "But let me pick a ball now!" His mom searched and searched for a long time, but there was a new toy. Tom loved to dig it up with his new magnet.
```

**top-k 40**, sample 1 (seed 1337, 91 tokens, stopped at <EOS>, repetition 0.0%):

```
Tom picked up the red ball and put it in his hand. He was happy that the ball belonged to the park. He gave it a hug and said, "This is the best ball of all the park". He said, "Wow, this picture is the good picture. I am glad you liked it." 
Tim smiled and waved goodbye to the red ball and went home. He felt happy and his new friend. He felt proud of his new ball and his own ball.
```

**top-k 40**, sample 2 (seed 1338, 37 tokens, stopped at <EOS>, repetition 0.0%):

```
Tom picked up the red ball and ran around the yard. He was feeling the rip of the ball and he loved to pretend he was the best. He was glad he could help the red ball play with them.
```

**top-k 40**, sample 3 (seed 1339, 53 tokens, stopped at <EOS>, repetition 0.0%):

```
Tom picked up the red ball and ran inside. His mom saw him playing with him. "Oh, Tom," he said. "But let me pick a ball now!" His mom replied, "There you get all your favorite food outside. Doothone in a piece of it, Tom."
```

**top-p 0.9**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 14.3%):

```
Tom picked up the red ball and put it in his hand. He was happy that the ball belonged to the park. He gave it a hug and said, "This is the best ball of all the park". He said, "Wow, this is my best ball ever!" He gave it to Tim. Tim smiled and said, "I don't want to go on the slide. It was fun and cool. Here, go, take it, let it go now."
Tim was sad and said, "OK, let's do it. It's still nice." He said, "OK, let's do it!" He took the blue ball and ran down the slide.
Tim's sister put the ball in a box and then gave it to Tim. She said
```

**top-p 0.9**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 12.9%):

```
Tom picked up the red ball and ran around the yard. He was feeling the rip of the ball and squeezed it.
"Look, Mom!" Tom said, holding the red ball.
"Let's dig it!" Tom said, grabbing his red ball and threw it back. The ball bounced on the fence and rolled it to the water.
"Stop, stop!" Tom said, holding the ball.
But the ball was too fast. The ball fell on the grass and hit Tom's face. He started to cry.
"Ow, that!" Tom said, holding the ball.
"Oh no!" Tom said, crying. "Mom, the ball is too fast. It is lying on the grass. It hurts our faces."

```

**top-p 0.9**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 4.1%):

```
Tom picked up the red ball and ran inside. His mom saw him playing with him. She yelled at Tom and yelled, but Tom screamed and ran away.
She was too busy playing and Tom saw a big sign on the ground. It was a sign that said "Dian" over and over. He said it said "hant. I did not have a dog. Is it for a while. Here it go". Tom was upset and started to cry.
Tom was mad and sad. He said "No, I want the ball. It made a big mistake. You have to be careful. Go away and not do it again."
Tom felt sorry and ashamed. He did not want to see his mom was mad. He did not want to make
```

### `lily_key` — prompt: `Lily found a shiny key under the bed. She`

**greedy**, sample 1 (deterministic, 150 tokens, hit the cap, repetition 82.3%):

```
Lily found a shiny key under the bed. She was very happy. She hugged her mom and said, "I love you, mom. I love you."
Her mom smiled and said, "I love you too, Lily. You are a very good girl. You are a good girl. You are a very good girl."
Lily smiled and said, "I love you, mom. I love you. I love you."
Her mom said, "I love you too, Lily. You are a good girl. You are a good girl. I love you."
Lily said, "I love you too, mom. I love you. I love you."
Her mom said, "I love you too, Lily. You are a good girl. You are a good
```

**temp 0.7**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 3.4%):

```
Lily found a shiny key under the bed. She picked it up and put it on her head. She put it in her pocket and smiled.
"Look, Lily, can I have this one?" she asked.
"Yes, Lily, but be careful. Don't touch it. It's too hard," her mom said.
Lily was happy. She liked to go outside and have fun. She liked to run and jump in the sun. She liked to go inside and play on the swings. She noticed that the grass was not the first one. She wondered if there was a stone. She climbed on the slide and slid down. She saw the stone on the ground and it was red and bent down.
"Hey, that's mine!" Lily said.
"No
```

**temp 0.7**, sample 2 (seed 1338, 87 tokens, stopped at <EOS>, repetition 0.0%):

```
Lily found a shiny key under the bed. She was so happy and couldn't be lost anymore. She said, "Mommy, can I have this key?" Mommy said, "Yes, they are. Let's play with it."
They played together all day. Lily was happy too. She could not find her toy car. She was very happy and she hugged her mommy. She was happy she got her toy car and she couldn't wait to show it to her mommy.
```

**temp 0.7**, sample 3 (seed 1339, 138 tokens, stopped at <EOS>, repetition 3.0%):

```
Lily found a shiny key under the bed. She picked it up and showed it to her dad. She told him it was a lawyer.
"Wow, a lawyer showed you to play with me!" Timmy said. He was very excited to see the lawyer and make up stories.
He ran to the door and waited for it to start his dad. After a while, Timmy went to the doctor and said goodbye to his dad. He took off his razer and put it on his face.
When Timmy reached his dad's seat, he felt much better. He thanked his dad and went to bed with a smile on his face. He couldn't wait to tell his dad about it.
```

**temp 1.0**, sample 1 (seed 1337, 66 tokens, stopped at <EOS>, repetition 3.2%):

```
Lily found a shiny key under the bed. She picked it up with her mom and told her what the key belonged to her own. Lily accepted the key and brought it back.
Mom was very happy that her brother accepted the key. They spent the decideing things in the world and had to understand that protecting the key to remember the key keys inside.
```

**temp 1.0**, sample 2 (seed 1338, 124 tokens, stopped at <EOS>, repetition 3.3%):

```
Lily found a shiny key under the bed. She decided to use the key and be working in the right spot for the treasure. She found a bike and bent down to introduced it with her.
"Here your car, do you know," Lily said. "We were so jolly on the necklace."
"Yes, we did!" Lily said. "We made a new doll, a thought I'd never met it again!" The car smiled and said it was her friend.
Lily held on tight and continued her walk home. She knew that she would help her dolls go now in the park every day and this time she got lost.
```

**temp 1.0**, sample 3 (seed 1339, 123 tokens, stopped at <EOS>, repetition 0.0%):

```
Lily found a shiny key under the bed. She picked it up in her hand, but something foolish happened. The priel came back and she was very sad! She asked her mom where she found her tuly guess.
"I wish I had a friend in my room, my package. Do you know what you said, but I have saw it a new one. It could leave," her mom said, pointing at a long world outside, and showing the light in Lily.
Lily was so excited! She opened the new spaceship and couldn't believe her luck! She called it out "MVacot Try!"
```

**top-k 40**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 1.4%):

```
Lily found a shiny key under the bed. She picked it up with her mom and told her what the key belonged to her mom. Lily was very happy to see the key unlocked. They looked everywhere but Lily could not find her mom.
They looked all around. Grandma saw the key and said, "Why did it look?" Lily thought for a difficult key, but she was confused. She was scared of going to keep it.
She opened the key and saw a toy car on the shelf. It was so shiny and tight. Lily tried to open it, but the car was too tight. She cried out for help, but she couldn't.
Then, when she heard a voice. She said, "Stop! You need to open it, you know wherever it
```

**top-k 40**, sample 2 (seed 1338, 147 tokens, stopped at <EOS>, repetition 5.6%):

```
Lily found a shiny key under the bed. She decided to use the key and bendi. She said, "Look, Lily, this key is open!" Mom said, "Wow, Lily, that will be unlocked your toy box!" Lily thought it was a good girl. She wanted to open it. She ran to the garage and opened the key. Inside were all toys, dolls, balls and coins. Then she ran to the attic to the open room. She was very excited.
Lily was so happy. She said, "This is yours! I want it for my bike now!" Mom said, "Let's pick a new one." Lily smiled and felt very good. She was glad she had her new toy. She knew mom would help her.
```

**top-k 40**, sample 3 (seed 1339, 83 tokens, stopped at <EOS>, repetition 2.5%):

```
Lily found a shiny key under the bed. She picked it up in her hand, but something foolish happened. The priel felt bad and she didn't get hurt. Her mom was very upset and decided to help her daughter. When they finally folded a new toy and Lily was very happy. She hugged her mom and thanked her for helping her daughter. They finished folding the toy and played together again. Lily felt happy that she could help her daughter.
```

**top-p 0.9**, sample 1 (seed 1337, 137 tokens, stopped at <EOS>, repetition 1.5%):

```
Lily found a shiny key under the bed. She picked it up and put it on her crown. It was a magical key. Lily was very proud of herself. She didn't want to open it. But her mommy had told her that the key might be used for a lot of coins. She didn't understand why the key was fragile.
After her mommy's story, Lily was so happy. She remembered that it would be probably like her mommy. From that day on, she made sure to keep her room close. She learned that it is a special thing and it won't be afraid if that she looks just as fondly, and that it's important to keep it as things for you.
```

**top-p 0.9**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 2.7%):

```
Lily found a shiny key under the bed. She decided to use the key and be working in the right spot for the treasure. She found a bike and went inside.
"Be careful, Lily," she said. "Let's dig and wash our hands."
She dug a hole and scrubbed the key. The key could not get dirty. It was broken. Lily's mom picked the key and ran to find out the key. She tried to open the key but it couldn't.
"Where are you?" she asked her mom. "I think we can help, mommy."
"I can open the key," said her mom. "We can find the keys in the gate."
Lily was scared. She was impatient. She held the key and ran back
```

**top-p 0.9**, sample 3 (seed 1339, 128 tokens, stopped at <EOS>, repetition 0.0%):

```
Lily found a shiny key under the bed. She picked it up and hugged it tight.
"Look, Lily, look!" She said. "It's a big, scary monster!"
"I know, Lily, but I saw it first," Lily said. "I'm hiding under the blanket."
They turned the key and said, "Good idea, Lily. Let's try to be friends."
They ran to the castle and played together. They had fun and laughed together. Lily was happy and sleepy. She said, "But I want to keep the sun, Mom!"
They had fun playing games and laughing. They pretended it was a superhero.
```

### `rainy_friends` — prompt: `It was raining outside, so the two friends stayed inside and`

**greedy**, sample 1 (deterministic, 28 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and watched the rain.
The rain was so happy that it was safe and they stayed inside for a while, and the rain was safe.
```

**temp 0.7**, sample 1 (seed 1337, 6 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and watched their dry sunset.
```

**temp 0.7**, sample 2 (seed 1338, 41 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and watched the sun setting and gazed up. The little girl was happy and she had the warm sun. She had such a wonderful day and she was ready to go inside and enjoy her sun.
```

**temp 0.7**, sample 3 (seed 1339, 93 tokens, stopped at <EOS>, repetition 4.4%):

```
It was raining outside, so the two friends stayed inside and looked inside for a warm warm, tight.
The bear was sure he should always make sure it was warm and safe, so he was careful. He carefully picked up the blanket and put it on the sun, so it warm and warmed it.
The bear was excited and he quickly started to close his eyes and realised it was the perfect warm and cozy room. He smiled, knowing that if he was a safe place, it felt safe and warm.
```

**temp 1.0**, sample 1 (seed 1337, 88 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and watched their dry stars.
Mong the story is what takes when something is scared, and it was also a big, filthy. The two of them talked about it if it was still as bounged at the rain.
Moration and the two friends had fun playing together. They learned new things and promised to be gentle and protect them. And they all lived there, it also as the rain was beautiful and sunny again.
```

**temp 1.0**, sample 2 (seed 1338, 93 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and their house often night in the park. Their parents laughed and said to the warm sun and warned the golf course. The house was clean and wet, so they lit up each other nicely.
The music waved and everyone waded and the on course they couldn't wait to get back. Everyone gathered around the dry to relax, happy just the rest of their siblife times had to come back when it came out of the sky.
```

**temp 1.0**, sample 3 (seed 1339, 51 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and looked inside for a warm warm, tight tight to dry and dry herself back. She carefully leaned against the window and gently stepped back inside. The jack tammed down and settled down, outside, enjoying the cold warm warm sunshine and warm.
```

**top-k 40**, sample 1 (seed 1337, 85 tokens, stopped at <EOS>, repetition 2.4%):

```
It was raining outside, so the two friends stayed inside and watched their friends. They ran outside to look for the det to cool down. When they were inside, the big, gray light made of blue. That was the family and they were so happy.
The two friends had lots of fun playing together! The little ones were so happy to have new friends. They even remembered to share something with their friends. They had found fun together and had the best day ever!
```

**top-k 40**, sample 2 (seed 1338, 74 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and their house, running in and playing nicely.
Suddenly the rain started to go and the rain came down. The two friends were playing and they all had a fun day. Then they returned each lural and brought a new toy.
The sun on the umbrella, and everyone smiled. Then they sat down to play in the rain, happy just the rain.
```

**top-k 40**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 4.1%):

```
It was raining outside, so the two friends stayed inside and looked inside for a warm warm, so they could dry and dry herself back. She said to them, "Let's have some cookies!" Everyone smiled and said they were good. Bigi, the little girl said, "Let's look around the park and have fun." So they all went to the park to find some fresh snacks to dry. The little girl was happy that the tree was able to dry leaves. Her friend, the girl, went to have some cookies and drink.
But there was a big dog running around. The little girl wasn't so much, she was still chasing the dog. She said to her friend, "Can I sled home now?"
Her friend was mad, but she couldn't. So
```

**top-p 0.9**, sample 1 (seed 1337, 27 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and watched their dry stars.
Mong the story is what a wonderful day, but when it's a lot of years old can.
```

**top-p 0.9**, sample 2 (seed 1338, 16 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and their house often night sky and gazed as they drove outside to sleep.
```

**top-p 0.9**, sample 3 (seed 1339, 61 tokens, stopped at <EOS>, repetition 0.0%):

```
It was raining outside, so the two friends stayed inside and looked inside for a warm warm, tight tight to dry and dry herself back. She carefully leaned against the window and gently stepped back inside. The two friends were standing at the window, holding hands and shining brightly. They were warm and cuddling together, until one time it was nice and sunny.
```

### `cat_window` — prompt: `The old cat sat by the window watching the birds. Suddenly,`

**greedy**, sample 1 (deterministic, 57 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the cat heard a loud noise. It was a big dog. The cat was barking and looked scared. The cat ran away and the cat ran away.
The cat was happy to be safe. The cat and the cat were safe and happy. They had a new friend.
```

**temp 0.7**, sample 1 (seed 1337, 131 tokens, stopped at <EOS>, repetition 14.1%):

```
The old cat sat by the window watching the birds. Suddenly, the cat pointed out to the window. The cat jumped up and said, "Hello, little one!"
The cat said, "Little mouse, why don't we get too close?" The cat replied, "I can help you!"
The cat replied, "I think I can help you!" The cat smiled and said, "I can help you. Look down there, it can help you get out." 
The cat was happy and said, "Thank you, sir. I believe you can help you." The cat thanked the cat and said, "You're welcome!" The cat and the cat had a fun day together.
```

**temp 0.7**, sample 2 (seed 1338, 44 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the window became very loud and gazed. The cat had to go inside and found a big, white cat. The cat was very quiet and followed the cat. They were so happy and had a new friend.
```

**temp 0.7**, sample 3 (seed 1339, 119 tokens, stopped at <EOS>, repetition 7.8%):

```
The old cat sat by the window watching the birds. Suddenly, the boy heard a loud noise.
The tiger told the cat, "What happened, little one?"
The little dog was surprised and looked up at the tiger. The little mouse said, "I'm sorry, but I'm here to make the tiger free!"
The little dog thanked the little cat and said, "Let's try to be free! We can be free!"
The little dog was happy to have saved the tiger. He was grateful and thanked the little dog for helping him. From that day on, the tiger and the little dog were the best of friends.
```

**temp 1.0**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the shield flames appeared. It was the house! It turned red and Jim scared the old man.
"What are you doing in the house?" asked Grandma.
“I'm hiding,” said Julens. Grandma pointed to the man. He said he was playing he decided to remember where he had seen the door burned so much.
Julens his neighbour itch wanted to destroy the house! He ran around it and got upset. He noticed that the old man was grinking things. 
"Oh, why did you destroy the house?" he asked.
He was sad, and he was too scared to separate. 
Juluctation was a 3-year-old little stop
```

**temp 1.0**, sample 2 (seed 1338, 126 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the window became gentle in and gazed as they saw a pond. They were pointing to the ducks and decided to introduced them to a friendly calmfish. The little bird said they were watching him swimming in the waves. The cat waved goodbye and left them with a gentle smile.
When the waror to the two enjoyed the surf and thought about yogi the world. He had the best day when it came out of the sky and held on tight abstic. Everyone cheered and laughed as the boy followed the rin proud of them. Together, they promised to join a moment in new game.
```

**temp 1.0**, sample 3 (seed 1339, 95 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the boy's fur got down, and he felt a little sad. A bee flew into the sky and looked up at him. The boy showed him how to hold the cheek laumb against his fur outside. He said, "Yuck! Obirping hurts” the old cat said, "eyning is good and live here to leave. The boy was still on the side of the house, feeling much less tired and so he continued to explore the world.
```

**top-k 40**, sample 1 (seed 1337, 98 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the cat pointed out to its owner. The cat jumped up and barked at the bird. The cat was very surprised and didn't know what to do. But the window had already gone. The cat watched as the cat had the back of the tree. They both thanked the cat because they were glad to meet it again. The old man smiled at them and happily sat down to watch it. The cat stayed on the branch forever and always kept on their adventure while the bird walked away.
```

**top-k 40**, sample 2 (seed 1338, 70 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the window became full of sunshine and came out. The cat had to go inside and found a big, white cat. The cat was terrifire and snatched close to see the wild cat. 
The cat thanked the cat and ran back home. From then on, the cat never stopped hoping to pet the cute little furry cat.
```

**top-k 40**, sample 3 (seed 1339, 144 tokens, stopped at <EOS>, repetition 2.8%):

```
The old cat sat by the window watching the birds. Suddenly, the boy's fur got cold, so he felt a hot day. Answer thanked the old cat and asked him why he was feeling uncomfortable. Another bird said, "I'm here a new friend and I'm here to rest. Do you know why you need it so good?" 
The little bird said, "It's okay! I'm hungry too!" He said, but slowly approached the bird. He said, "Welcome to me and I should have some food. Then you are ready."
The little bird quickly flew away from the little bird and thanked it for saving them. Worm was happy that he could help and he soon fell asleep.
```

**top-p 0.9**, sample 1 (seed 1337, 55 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the cat pointed out to the big, orange window. It jumped and laughed a happy music. The cat said goodbye and drove away. The old cat was happy that it had a new friend. From that day on, the old man learned to always skip around the window.
```

**top-p 0.9**, sample 2 (seed 1338, 104 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the window became full of sunshine and came out. The lady said to the rabbit, "Look, go now!" The boy was excited and they both ran to their friend, asking if the bird said. They went down the window and carefully knocked on the door. The door was empty, and the bird perched in the water. The two birds were relieved and they stayed on the the window. The bird was still happy and stayed safe. They learned that the park was even more kind and what we can happen.
```

**top-p 0.9**, sample 3 (seed 1339, 57 tokens, stopped at <EOS>, repetition 0.0%):

```
The old cat sat by the window watching the birds. Suddenly, the boy's fur got down, and he felt a little sad. He started to cry.
The dog whispered goodbye to the little boy and walked away. The child saw him but then understood what he had told him. He had been thankful for the chance to try new things.
```

### `park_dialogue` — prompt: `Mom said, "We are going to the park today!" Ben was so happy that he`

**greedy**, sample 1 (deterministic, 42 tokens, stopped at <EOS>, repetition 25.6%):

```
Mom said, "We are going to the park today!" Ben was so happy that he had a new friend. He said, "I'm glad you found a new friend!"
They went to the park and played together. They had a lot of fun. They had a lot of fun.
```

**temp 0.7**, sample 1 (seed 1337, 31 tokens, stopped at <EOS>, repetition 0.0%):

```
Mom said, "We are going to the park today!" Ben was so happy that he had saved his family.
They played on the swings and the slide, and the slide. They laughed and laughed. They had a lot of fun.
```

**temp 0.7**, sample 2 (seed 1338, 16 tokens, stopped at <EOS>, repetition 0.0%):

```
Mom said, "We are going to the park today!" Ben was so happy that he got on the swings and slides. He was so happy to have a friend.
```

**temp 0.7**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 5.4%):

```
Mom said, "We are going to the park today!" Ben was so happy that he jumped up and down. They played on the swing and had fun. But then, Ben said, "No! This is mine now!" His mom said, "No, Ben. That is a bad idea, but we need to find it."
Ben was sad. He said, "But I saw it first. It looks much better." His mom said, "It's a good idea, but you need to be careful. The slides can fall and have fun." They looked up and saw the slide and the park. They saw a slide and a swing. They ran to the swing and climbed up the ladder. They climbed up and slid down. They laughed and had fun.
They did not want to swing anymore. They looked
```

**temp 1.0**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 0.0%):

```
Mom said, "We are going to the park today!" Ben was so happy that he had saved listening to the big kids. He said to his mom, "That was a good gift for you. Can you read it now?" His mom smiled and said, "Yes, you can. But you decide to show it tonight?" Ben said yes and they sat down to read. They started to read the whole voice and admired the amazing things it might hear his heal it.
Afterwards was finished, Lily and Ben sat down together and read the sweet bear's someday. Together they planned "Be good, Ben!" They picked up the soft bear and pretended to be brave and grateful to Lily for their idea. Ben trusted his grandma and returned safely. He even had a great time and Lily loved reading books
```

**temp 1.0**, sample 2 (seed 1338, 93 tokens, stopped at <EOS>, repetition 4.4%):

```
Mom said, "We are going to the park today!" Ben was so happy that he got on the bike in and drove to the park. He proved it with his mom.
Mom said, "Look, Lily, they are so excited. You have to go on the ride. It looks very calm."
Lily was so excited on the ride. She pointed to the park. It was a beautiful ride to slide, so she got slid down and slid down the slide. She had so much fun when it was time to go home.
```

**temp 1.0**, sample 3 (seed 1339, 144 tokens, stopped at <EOS>, repetition 0.0%):

```
Mom said, "We are going to the park today!" Ben was so happy that he jumped up and down. They played tag, swing and swing. The park was the fastest on stage. Ben laughed a lot and felt better.
But then, a big time, there was a little boy who went in the carhip. The dog was spoiled and Max said, "I have saw it a new song. It could leave here." Ben thought it was sad, but he learned his lesson. So, he played the music he went near the stage and went back to playing on the stage. After the day, he went back to the stage playing with his new song. Max was happy that he had been kind to let Worm come to telling the way to see his own jail.
```

**top-k 40**, sample 1 (seed 1337, 144 tokens, stopped at <EOS>, repetition 0.0%):

```
Mom said, "We are going to the park today!" Ben was so happy that he had saved his adventure.
After Timmy's house, he asked his mom if he could go. His mom said yes, but only if his dad was there for him if he couldn go back.
Jack was very sad. He didn't want to leave but he knew it was dangerous for him. But his mom suggested that it was dangerous. They got lost and had to keep it in their pocket safely. 
Jack was happy again and got his family back home. He had found his mommy and they were very happy that he was back home and now, he had to remember to say sorry to his dad and that it's important to stay there and help him find out for his family to the park.
```

**top-k 40**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 2.7%):

```
Mom said, "We are going to the park today!" Ben was so happy that he got on the bike in and drove to the park. He had to go to the park, who also went there. The park was very quiet and the toys were playing. Then, the park had a new ball.
"Max, you have a nice ball," Lily said. "I did it big and fun!"
They were playing some more, but they loved their new ball. A group of kids came and said, "Hello, kids!"
The kids waved at them and hugged their new toys. They said, "Hello!"
"Yes, we're here!"
The kids said, "We are playing together. Let's play with your ball and my ball!"
They laughed and ran and pushed each other around.
```

**top-k 40**, sample 3 (seed 1339, 112 tokens, stopped at <EOS>, repetition 1.8%):

```
Mom said, "We are going to the park today!" Ben was so happy that he jumped up and down. They played tag, swing and swing. The park was the fastest one ever. Ben laughed a lot and felt much better at the park.
When the park stopped, the sun set and the slides started to rain. Ben loved to play on the swings and slide like his ball. The park was full of sun and the sun. Ben had lots of fun until it was time to go home, and Lily sat down on the boo-boo while playing. She would have a flashlight back and take a bath time.
```

**top-p 0.9**, sample 1 (seed 1337, 41 tokens, stopped at <EOS>, repetition 0.0%):

```
Mom said, "We are going to the park today!" Ben was so happy that he had saved his adventure.
Afterwards, the sun began to set and the park was lots of fun. Ben was so proud of himself. He was happy that he had helped all of the park.
```

**top-p 0.9**, sample 2 (seed 1338, 112 tokens, stopped at <EOS>, repetition 1.8%):

```
Mom said, "We are going to the park today!" Ben was so happy that he got on the bike and played on the swings. He said, "Look, Lily, this bike is so pretty!" The man laughed and said, "Yes, he was very cool. Let's play on it."
As they played, the man said, "That's a great idea, Ben!" They sat on the swing and played on the swing. They made funny faces with their faces and faces. Lily was happy too. They said, "Thank you, mister!" They hugged each other and thanked him for being such a good friend.
```

**top-p 0.9**, sample 3 (seed 1339, 127 tokens, stopped at <EOS>, repetition 0.0%):

```
Mom said, "We are going to the park today!" Ben was so happy that he jumped up and down. They played tag, swing and swing. The park was the fastest one ever. Ben laughed and pointed at it. It was a lot of fun. He was getting thirsty outside and sat down and fell down. He threw his hat back and went to the slide and slid down.
"Wow, this day!" Ben said, holding his hat on his head. "What fun!" Ben smiled and said, "Let's play a game!"
They played for a while and then went home. They put on their shoes and started to clean them. They were tired but then decided to go home.
```

### `robot_wish` — prompt: `Once upon a time, there was a tiny robot who wanted to`

**greedy**, sample 1 (deterministic, 150 tokens, hit the cap, repetition 41.5%):

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He asked his friends if they could go on an adventure.
His friends said yes, but they were too little to go on an adventure. They went on a journey and found a big, shiny object.
The little robot was so excited to explore the world. He wanted to explore the world, so he started to explore.
He found a big, shiny object and he was so excited. He wanted to explore it. He carefully picked it up and started to explore.
He found a big, shiny object and he was so excited. He couldn't wait to explore the world around him.
He was so excited to explore the world around him. He had never seen anything so shiny before. He was
```

**temp 0.7**, sample 1 (seed 1337, 88 tokens, stopped at <EOS>, repetition 9.4%):

```
Once upon a time, there was a tiny robot who wanted to give it a special gift. He was so excited to give it to the robot. He accepted the robot and brought it to the boy. The boy was so excited!
The robot accepted the robot and he thanked the boy. The boy's wish and he had it. The boy smiled and waved goodbye as he went home.
The boy was very happy and his robot it happily home. He was no longer famous again!
```

**temp 0.7**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 4.1%):

```
Once upon a time, there was a tiny robot who wanted to go on the gym. He was very excited to explore the gym. 
One day, the robot went on a journey. He was so excited that he started to get through the gym. It was so big that he couldn't get into trouble. He couldn't believe his eyes. 
But then, the robot shouted to his friends. They all wanted to learn and tell the robot that he hadn't waste it! The robot was so angry and shouted, "No! I want to stay inside! I want to go!" 
The robot was scared and started to cry. He said, "I want that prize!" 
The robot just shook his head and said, "No, I want more prize! I
```

**temp 0.7**, sample 3 (seed 1339, 147 tokens, stopped at <EOS>, repetition 11.1%):

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He went to the river to get some fish, but the robot was too short. He tried to make it move, but it was too high for him. 
The robot was sad but he tried to find a way to go down the river. He got so careful not to rock anymore! He was determined to get the fish! 
The little robot was so happy that he could help the robot. He worked hard and made it to the end of the river. The little robot was so proud of himself for helping the robot. 
The little robot was thankful for his friend, and he was very happy. He was happy that he could help and he was able to help the robot.
```

**temp 1.0**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 0.0%):

```
Once upon a time, there was a tiny robot who wanted to give it a special gift. He carefully collected the daisy and led him to a beautiful gift for all the joy. He showed it his favorite animal of charming shapes and leaves they found as many colors and patterns.
But soon, the robot couldn't find out that he decided to take something that difficult he wanted. He smiled admired his work and felt happy once there. it put the lariest animal in a nearby tree and got upset. He noticed that it simple in the tree plan and decided to keep the message.
He eventually caught the creature for a while, and could not make it it when it was okay. The robot decided to take the message. He was feeling happy again and couldn't forget about
```

**temp 1.0**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 4.1%):

```
Once upon a time, there was a tiny robot who wanted to go on the gym. He was feeling ign as if the robot was taking over and tried to take go home. 
One day, they decided to take their bags and go on the gym. The robot had a special missile!
He walked on the least paper: "What did you do to take! I to pay for something hot?" he thought for a moment the robot. He had an idea!
He came out of the gym and held his hand. The robot carefully to see if he was standing still watching. The next day he stood up and asked if he wanted a look. The robot smiled and said yes.
The robot was happy and he couldn't wait to get back on an adventure. The robot smiled
```

**temp 1.0**, sample 3 (seed 1339, 138 tokens, stopped at <EOS>, repetition 5.2%):

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He went to the river to look up and down the trees. 
The robot whispered goodbye to the robot, then walked to play. He saw some fish, fruits, but he was enough for him. He was very happy he had someone to meet someone in the river. 
The robot was determined to try and be friends. He had lots of fun doing his journey but all he was ignoring it. 
The robot and people loaded the river and then rolled back home. He was so excited as he rocked into the river, hallowed and closed his eyes, rock from feeling thick full of clouds. 
The end.
```

**top-k 40**, sample 1 (seed 1337, 111 tokens, stopped at <EOS>, repetition 0.0%):

```
Once upon a time, there was a tiny robot who wanted to give it a special gift. He carefully took the bow to the other robots carefully and gave it a special bag. The robot asked the robot if they talked to it again and they said yes. They made the robot even more happy. The robot's friends were happy.
The little robot thanked the boy and promised that it was the right thing. The little robot liked his new bow and wanted to finish himself, and he was happy he could help his friends. From that day on, every day he went on adventures and loved his new robot.
```

**top-k 40**, sample 2 (seed 1338, 95 tokens, stopped at <EOS>, repetition 0.0%):

```
Once upon a time, there was a tiny robot who wanted to go on the way. He was feeling very brave and said, "I can help you. You go slow!" 
The robot was so brave that he started to get rid of the trees. Suddenly, he felt a bit dizzy. He stopped and stepped back up to his little boy. 
The little boy smiled. He had two new friends, a big bear and a squirrel full of fun. They were so happy to help they go on on the road!
```

**top-k 40**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 2.7%):

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He went to the river to look up and down the trees. 
The robot and his brother went home together. The robot's parents saw him and said, "Ew, these things are for your own way to explore!" 
He was so careful so he waited and waited to find out why it went so far away. When he came back, he said, "Welcome in this forest!" His parents said, "Yes, let me explore this path. I hope we look around.”
The robot smiled and said, "Okay, I will always stay here all night you need. Let's go." 
The robot's dad smiled and agreed. They had a great day and when
```

**top-p 0.9**, sample 1 (seed 1337, 91 tokens, stopped at <EOS>, repetition 4.5%):

```
Once upon a time, there was a tiny robot who wanted to give it a special gift. He carefully collected the daisy and took it home. He placed it on the paper and started to feel very comfortable. 
He had a great idea and decided to take the picture to show his friends. He was excited to have some tape and decided to take it home. He packed it and walked home. His friends happily skipped away with him, happily, knowing that his good experience could help him.
```

**top-p 0.9**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 0.0%):

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He was feeling adventurous and happy. 
One day, while searching for a big adventure, he decided to go on a journey. He traveled and walked until he came across a tunnel. The robot was small and could hear the robot sing along. 
He moved into the tunnel and then to his two legs the shadows. As he started to tell himself that it was his way to go! The robot clapped his hands and clapped his hands. The robot made his body go back and forth made sure that no one had ever helped him. 
The robot's little new plan and he kept it everywhere until the stars around the woods together. The little robot was very sad and he wanted to go back
```

**top-p 0.9**, sample 3 (seed 1339, 100 tokens, stopped at <EOS>, repetition 4.1%):

```
Once upon a time, there was a tiny robot who wanted to go on an adventure. He went to the river to look up and down the trees. 
The robot and his brother happily wandered through the water and he saw many things on the ground. He saw new things and promised to find new friends. So the robot went over to play on the wide river and it sounded like a magic robot. 
The robot had lots of fun until the sun went down and the river went off. He waved goodbye and felt very content. The end.
```

## Checkpoint 5k (`checkpoints/transformer_best.pt`, step 5,000)

### `tom_ball` — prompt: `Tom picked up the red ball and`

**greedy**, sample 1 (deterministic, 150 tokens, hit the cap, repetition 85.0%):

```
Tom picked up the red ball and threw it at the ball. He threw it at the ball and ran to the ball. He ran to the ball and threw it at the ball. He ran to the ball and threw it at the ball. He ran to the ball and tried to catch it.
Tom and Lily were scared. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did not know what to do. They did
```

**temp 0.7**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 15.0%):

```
Tom picked up the red ball and put it in the ball. It was orange and made a big mess.
"Look, Tom, we found a red ball!" Lily said. "We can weigh our balls!"
"OK, Tom, let's play!" Tom said. He took the ball and threw it at the ball. He threw it at the ball with the ball. It made a loud noise it.
"Hey, that's my ball!" Tom said. He took the ball from Lily's mom. She tried to catch it. But the ball was too fast.
"Help, help!" Tom cried. He saw his ball stuck in the ball.
"Help! Help!" he shouted. He pushed the ball from Lily. She tried
```

**temp 0.7**, sample 2 (seed 1338, 116 tokens, stopped at <EOS>, repetition 5.3%):

```
Tom picked up the red ball and threw it around the sky and it flew away. He said, "Look, Lily, this bike is so pretty! Can I see it?"
Lily was amazed by how she looked at the tire. She picked it up and showed it to her mom. "Look, Mommy, I found it. It's so pretty!" she said, showing the tire more.
Her mom smiled and said, "That's so good, Lily. It's so pretty and pretty. Let's do it."
Lily nodded and said, "Okay, let's play with it."
```

**temp 0.7**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 16.3%):

```
Tom picked up the red ball and threw it out in the air.
"Wow, look! The ball is too fast!" Tom said.
"Me too," Tom said. "But we have to clean up."
They grabbed their new ball and ran to the car. The car hit the ball and made a lot of dust and dust. He was not happy.
"Wow, we did it!" Tom said. "You can break the ball!"
"Good job, Tom and Tom," Tom said. "But I have some water. The water are wet."
"You are brave," Tom said. "You can play with you. You are wet. You are a good boy."
"OK, Tom," Tom said. "But you have
```

**temp 1.0**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 5.4%):

```
Tom picked up the red ball and put it in the colorful ball. He tried to boil the ground, but the ball made it shines. He didn't win.
Tom was sad. He wanted new toys. He wanted cars too. He had an idea. He took my ball and ran out of the ball for a balloon. He rolled the ball and smiled.
Tom was kind. He found Tom it. He felt himself like a balloon. He could trade with his chain. He ran to the slide. He tried to catch it. But the ball was too high.
"Help, help!" Tom cried. He wanted to play with the ball too.
"Help! You could not escape. The ball is mine. This ball is mine!"
```

**temp 1.0**, sample 2 (seed 1338, 117 tokens, stopped at <EOS>, repetition 0.0%):

```
Tom picked up the red ball and sat on the couch in and becinated. He felt proud of himself and took a bike golf.
Mritooral cheered, "Toy! Follow the toy basket!"
Tom wagged his tail and jumped on a branch. The car set out of the car and lifted the car. Tom laughed, happy with the new bicycle.
But the moral of the story is: when you miss safe outside. Wight in the yard, aband can come back to me with your friends. Friends can be very still come true.
```

**temp 1.0**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 0.0%):

```
Tom picked up the red ball and ran to play in the dirt, but it was hard and hard to dry the bracelet. Mommy showed it to Tom and asked, "What did you smell?"
Tom thought for a moment and then he became a friend. He let go of the zipper and found it on the ground. He was very happy and said, "Let's try this. But it is too heavy for you to get broken."
But the zipper still looked so heavy. It was not dusty. It was spilly. "Oh no, you are doing it. No bad. It's a big magnet! Can I come back rock this night?"
Tom did not like that. He made it with a grown-up and a teeth. He thought
```

**top-k 40**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 9.5%):

```
Tom picked up the red ball and put it in the ball. It was orange and made a big fall.
"Look, Tom, Tom, what you can win. It's too hard for we might fall." Lily cried.
Mom looked at the ball in the truck and said, "Why did it?"
Tom did not want to touch the ball. He looked at his hat. He liked his ball, but she did not like it. He thought it was not nice. He wanted his ball too. He picked it and went to play with it.
"Look, Mommy, my dog did not mean!" she shouted.
Tom did not want to play. He wanted his ball back with his ball. He said, "Mom, Lily, I see
```

**top-k 40**, sample 2 (seed 1338, 112 tokens, stopped at <EOS>, repetition 0.0%):

```
Tom picked up the red ball and rang the ball in and pulled it. "You can't get it!" Lily said.
It went away, but the ball was too short. Lily felt sad and alone with the heavy rock. She felt sorry for Tom and took the ball back on the ground. "It's okay, Tom. It's okay. No more bounces."
Tom thought for a moment. He had a long stick on his stick and a toolbox from his ball car. He learned to be careful with his toys. He made a loose of blocks.
```

**top-k 40**, sample 3 (seed 1339, 57 tokens, stopped at <EOS>, repetition 0.0%):

```
Tom picked up the red ball and ran to play in the dirt, but it was hard and hard to dry the ball. Mommy showed Tom how to hold a stick and put it under her hands.
Tom was happy to have a new red ball for his dad. He knew it was important to try new things.
```

**top-p 0.9**, sample 1 (seed 1337, 150 tokens, hit the cap, repetition 5.4%):

```
Tom picked up the red ball and put it in the colorful ball. He tried to make it look amazing, but it was not working.
Tom was still miserable. He wanted to show his mom. She wanted her mom to make a bad monster for him. She asked him what he did and he told him he shouldn't wear it.
Tom did not say anything. He was kind and gentle. He said he wanted to play with his toys. He said he was sorry too, but he did not wear his toys. He went to play with his toys and played with his friends.
But then, a big man appeared. He had a hat and a big mane. He was angry and loud. He saw the big man who was not happy. He
```

**top-p 0.9**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 0.0%):

```
Tom picked up the red ball and sat on the couch with him. He wrote about sticks, holes and trees.
"Look, Mom, I found a stick on the wall!" she said, pointing to the wall.
"That's a cool, Tom. You are so smart," Lily said, laughing with a smile. She touched the signs and the slide was too.
But then, she saw a big dog lying on the road. The dog came out of the hole and held on tight. The dog barked and barked.
"Help, help! The dog ran!" the dog barked. It was scared. It barked and jumped and ran away. The dog and jumped in the rock. It fell down and got wet and wet.
Tom and Lily
```

**top-p 0.9**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 6.8%):

```
Tom picked up the red ball and ran to play with it. He tried to make the ball hard, but the ball broke.
"No, Tom, no, it's my ball!" Tom shouted. He tried to put it on a new ball in his dad's hand.
Mom saw what happened. She hugged Tom and took the ball to the hole. She hugged Tom and said, "Don't worry, Tom. I will throw this car, and this ball. It is bleing me."
Tom and Mia stopped crying and looked at the broken ball. They saw what happened. They felt sorry and said, "We're sorry, mom. We will not cut your tape. We can share and be fine."
Mom took the ball and ran to
```

### `lily_key` — prompt: `Lily found a shiny key under the bed. She`

**greedy**, sample 1 (deterministic, 150 tokens, hit the cap, repetition 70.1%):

```
Lily found a shiny key under the bed. She thought it was a good idea. She picked it up and put it in her pocket. She put it in her pocket and went to the park. She saw a big tree with a sign that said "Do not open it. It is a big tree. It has a sign that says "Do not open it. It is a sign that says "Do not open it. It is a sign that says "Do not open. It is a sign that says "Do not open. It is not open. It is a sign that says "Do not open. It is a sign that says "Do not open. It is a sign that says "Do not open. It is not open. It is a sign that says "Do not
```

**temp 0.7**, sample 1 (seed 1337, 27 tokens, stopped at <EOS>, repetition 0.0%):

```
Lily found a shiny key under the bed. She put it in her pocket and showed it to her friends. She was very happy. Lily and her friends played with the car together.
```

**temp 0.7**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 1.4%):

```
Lily found a shiny key under the bed. She wanted to use the key and be her mom. She said, "Look, Lily, I found a big box in the box. I brought it for a key!"
Her mom smiled and said, "That is a key, but you have to rest. It can be a law, but not close to the door. It will be your way."
Lily nodded and said, "Okay, mom. I will go to the hospital. But I want to go home. The doctor will be ready. He will read it again."
The doctor said, "I will remember this. It's just a big, nice place. The door will be ready."
Lily and her mom went home. They dreamed of more adventures. She
```

**temp 0.7**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 6.8%):

```
Lily found a shiny key under the bed. She thought it was a present for her mom. But she knew it was a good idea. She wanted to open the door, but it was too heavy for her.
She opened the door and saw a little girl. She was a tear in the door. The chain was old and it was very heavy. It had no hole. The chain made a loud noise.
Lily's mom saw the broken chain. She was not angry. She was scared. She said, "Lily, what happened? I can't open the lock. I can't open it. Please, please. I won't open it. We can't open. Please, please."
Lily was confused. She thought for a moment. She said, "Please,
```

**temp 1.0**, sample 1 (seed 1337, 27 tokens, stopped at <EOS>, repetition 0.0%):

```
Lily found a shiny key under the bed. She put it in her hand and thanked her mom for being so careful to cool. From then on, she always slid more and more.
```

**temp 1.0**, sample 2 (seed 1338, 37 tokens, stopped at <EOS>, repetition 5.9%):

```
Lily found a shiny key under the bed. She wanted the scale to get and be her favorite way. She tried to save it and found it. She tried to close it, but it was too late. The scale was gone.
```

**temp 1.0**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 2.7%):

```
Lily found a shiny key under the bed. She looked inside and saw it was a tight lock. She wanted to keep it. She tried to help it open and light it was locked. She pushed her tambbed and pushed herself. She called for a new box.
"This will be scary!" Sara said. She said. "I have a tool to unlock this chain. But I don't know what to do!"
But her mom was upset. She said, "Lila, I don't like train. It is spike." She pushed her. She wrote on her mustce and gave her a tool.
Lila shook her head. She said, "No, mom. Come here. The key will see." She did not want to cry.
She opened
```

**top-k 40**, sample 1 (seed 1337, 32 tokens, stopped at <EOS>, repetition 0.0%):

```
Lily found a shiny key under the bed. She put it in her hand and held it tight. It was safe to her. Lily was very proud of herself. She knew that flying made her feel safe.
```

**top-k 40**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 6.8%):

```
Lily found a shiny key under the bed. She wanted the key to unlock her door. She ran to it and tried to open the key. But the key was locked. Lily was so sad. She hoped the key was locked.
The door door opened. It was locked. It locked, on the key. It could not open well. It locked the keys to unlock something. The keys unlocks the lock. It had locked the keys on. The door unlocks and left them. The keys opened. It unlohed through the lock.
Lily heard the door open. She opened. It opened. She saw her mom. She was in the keys on her hand. She cried in the house and held Mommy. She wished she
```

**top-k 40**, sample 3 (seed 1339, 150 tokens, hit the cap, repetition 0.0%):

```
Lily found a shiny key under the bed. She looked inside and saw it was a box. She put it in her backpack and waited. She climbed on the chair and opened the key. She pushed her tammed it.
But as she went, she saw a box on the window. She saw the box on the shelf. It was red and swirls and spits. She could not hear it on the floor. It was a big box. Her mom and dad did not hear the noise.
"No, Lily, I have some mail. The mail box is locked. No things come out, you will have enough money or things or you will. Go off now, Lily. That's not bad." Mom said.
"You are not a good boy
```

**top-p 0.9**, sample 1 (seed 1337, 33 tokens, stopped at <EOS>, repetition 0.0%):

```
Lily found a shiny key under the bed. She put it in her hand and thanked her mom for being so careful to cool. From then on, she always watched the light glow in the corner of the garden.
```

**top-p 0.9**, sample 2 (seed 1338, 150 tokens, hit the cap, repetition 1.4%):

```
Lily found a shiny key under the bed. She wanted to use the key and be her favorite way. She tried to save it and tried again. But the key was too high.
Lily's mom saw what happened. She came into the door. She saw the key in the chair. She was a little girl. She was lonely. She was not nice to keep her to pay for her toys.
She hugged her mom. She said, "Don't worry, I'm sorry. I won't hurt myself. The key is mine."
Her mom said, "It was dangerous. It is dangerous. It can hurt yourselves. Come on, let's go to bed. I will get on you down."
Lily stopped and listened to her mom. She did not
```

**top-p 0.9**, sample 3 (seed 1339, 147 tokens, stopped at <EOS>, repetition 5.6%):

```
Lily found a shiny key under the bed. She looked inside and saw the hole, but something felt magical. The prissible place broke.
Lily looked and saw a shiny gold coin on the ground. It was a long way to put it on a new box. She wondered what it was. It was someone else. It was so beautiful. It was a new place where it could leave. Lily hugged the coin and smiled. She said, "I'm glad we found this coin. It was blewoast, and it may be more treasure."
Her mom hugged her and said, "I'm glad you're welcome. Let's go home." They went to bed and cozy. They hugged and walked to her house. They felt cozy and cozy.
```

