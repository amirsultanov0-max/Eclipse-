# Stage 5.5 — structured error analysis

_2026-09-21 22:38_ · read-only, no training.

- checkpoint **6.1** (`checkpoints/transformer_6_1_best.pt`, step 9,750)
- decoding **top-k 40, temperature 1.0** — an evaluation configuration for this analysis, not a claim that it is the best strategy
- 8 prompts x 5 seeds (1337-1341) = **40 generations**, cap 150 tokens
- scoring rules were written and committed in `eval_error_analysis.py` **before** any generation was produced, and were not adjusted afterwards

## Category 1 — entity / name consistency (mechanical)

| outcome | count |
|---|---|
| first name recurs | 13 / 40 (32%) |
| first name changes to another | 4 / 40 (10%) |
| first name disappears | 3 / 40 (8%) |
| no proper noun at all | 20 / 40 (50%) |
| **at least one uninitiated new name** | **12 / 40 (30%)** |

## Category 3 — completion quality (mechanical)

| measure | count |
|---|---|
| reached `<EOS>` before the cap | 32 / 40 (80%) |
| ends on terminal punctuation | 32 / 40 (80%) |
| contains invented words | 7 / 40 (18%) |
| **grammatical (clean end AND no invented words)** | **27 / 40 (68%)** |
| mean 4-gram repetition | 1.4% |

## Category 2 — causal / story consistency

_Pending: scores not yet recorded in `eval/causal_scores.json`._

## Every generation with its scores

### `dog_hungry_1337`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it hopped away into the hole. It was the best meal.
The little mouse was very happy. With each bite, the cheese seemed to eat its meal. Then it had a delicious meal as a delicious snack. 
The chubby mouse thanked the mouse for helping her. They went to the next meal and it was full. Then the cat took a snack. Then it sat back at the table, ate it, and it ate until it was full. 
The little mouse ate the delicious meal and went back to sleep. As the bite grew brighter, the mouse had forgotten to grow up!
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 122 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 5.0%

### `dog_hungry_1338`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it decided to find some carrots and planted them. He said, "Look, I found a carrot, you found! The carrots are very tasty!" 
Tom was excited and said, "You're such a good friend." The dog ran and ate a little bite, until it was very empty. It was the best hiding spot to eat. 
From that day on, the dog and Tom were the best of friends and played together. They played in the yard every day and never felt guilty again. They were still friends forever.
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: none
- **cat 3**: 109 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `dog_hungry_1339`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it went inside and landed down down to the table to eat and eat it.
The dog was sad, but then a wise old man came by. He had a bad idea. He took the food outside and ate it in his tummy. He was so sad that someone else would come to help him.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 61 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `dog_hungry_1340`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it wanted to eat the meal. It bit bit the nose and said, "I want to eat it first!"
The dog tried to resist and said, "It's just a very quiet meal. I must have some more cheese and cookies. You should be patient and wait."
The man walked away, but the dog chased after it. The man never forgot the lesson.
The end.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 80 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.6%

### `dog_hungry_1341`

**Prompt:** `The dog was hungry, so it`

```
The dog was hungry, so it waited for the mouse to eat. The mouse looked at the mouse’s house and said "Let's make a new toy for the mouse!"
The mouse agreed and opened the hole to get to the mouse. The mouse was so surprised that it wanted to eat it, but it was also very sad.
Later, the mouse saw the mouse eating more pieces. It went away, making something very hot. The mouse felt bad for trying his food and didn't have eaten the mouse. They all hugged it and played with it together. From that day on, they were the best of friends and enjoyed taking the mouse.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 127 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `girl_store_1337`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because she lost her toy. When she arrived, the store was big enough for Lily. Lily and her mommy said goodbye and drove away. Lily was so happy that her mommy had helped her go back. They packed the present and gave Lily a surprise for her! The end.
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: none
- **cat 3**: 55 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `girl_store_1338`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because her mom bought some candy and put on them. She said, "Mommy, I found a candy on the shelf. The candy's very cool." 
After playing with her toys, her mom hugged her back. "It's done, sweetie, you're very brave when you put your candy. It's important to get to pay for something the candy." 
Millie smiled and said, "Yes, I'm brave! I hope from going in the fridge!" 
From that day on, Molly was careful when she was walking outside to play. As she read a few extraments, she became her regular exercises from the post. She loved her exercises and would never forget the day, but when she did her
```

- **cat 1**: first name `Millie` -> **changes**; uninitiated new names: ['Molly']
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: ['extraments']; grammatical: **False**; repetition 0.0%

### `girl_store_1339`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because she was very excited. She put the package on the shelf and ran back to the house.
When the day arrived, the girl was very excited to play. She ran around the house, enjoying the suns and exploring in the sunshine. After a while, the girl got very angry that she ran inside. The box was full of colorful things. The girl was very angry that the tree didn't mean all. She started to punish her mom and dad and dading the park.
But the girl always wanted the old tree. She was very excited to come back and start running away from the park.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 124 tokens, reached <EOS>; ends cleanly: True; invented words: ['dading']; grammatical: **False**; repetition 1.7%

### `girl_store_1340`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because she saw a lot of people who were trying to make it fit. She was confused and didn't know what to do. 
When the mommy was finished, she put the pieces together to the little girl back. The little girl got up the pieces and began to draw a picture on it. The little girl was very excited and she asked her mommy why it was so tasty. 
The little girl smiled and happily showed her mommy the piece. From that day on, the little girl was the biggest!
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 103 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.0%

### `girl_store_1341`

**Prompt:** `The girl went to the store because`

```
The girl went to the store because it was full of fun things. She found a big piece of cheese and ate it all up. She ate it all up and felt very special.
The girl took the cheese and enjoyed playing with it. It was a special treat that made her dream. She knew her wish would be more special.
Later that day, she went to the house to see what it was. She wondered which was so exciting. When she opened the door, she looked around and saw something she had never seen. She had picked it up on the shelf and smiled.
The girl thanked the magic. She smiled and hugged it close. From then on, she was very careful with the cheese before she left her secret room. As she knew it wasn't
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 2.7%

### `tom_ball_1337`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and put it in his hand. He was happy to have his new friend.
But then, he was not careful. He saw his friend, Lily. He had a yellow nose and a green shirt. He was very happy. He took the red ball and took it.
"Mom, I like your yellow ball!" Tom said, holding up.
He gave it to Tom. It was red and shiny. It had a long wire on it. Tom and Lily felt happy. They picked the red ball and threw it at the green ball. They picked it up.
Mom was proud of Tom and Lily. She was proud of them.
"See, Tom is amazing. You did a great job. This ball is very special
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: none
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 1.4%

### `tom_ball_1338`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and rang the ball in and barked nicely.
Tom said, "Look, Lily, I found a big yellow ball that was not ours. Look, I found a yellow crayon on it."
Lily picked up the ball and put it near Tom on the ground. "It's very pretty. Look, it's a pretty red bow. It can weigh it and tell the other stuff that don't belong to us."
Tom said, "Thank you, Lily. You are a good friend. You are a good friend."
Lily hugged Tom and said, "You are welcome, Tom. You have a nice pair. You can play with something new instead."
Tom and Lily played with the ball more time. They
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: ['Lily']
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 8.2%

### `tom_ball_1339`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and ran inside right in the hole.
Tom and Lily were very happy. They ran back to their house and played with a band-aid. Tom learned that it's good to have fun and to always clean up after playing in the dirt.
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: ['Lily']
- **cat 3**: 49 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `tom_ball_1340`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and put it on the ground. It was shiny and round. He was excited to show it to his mom.
His mom said, "Tom, you need to play, but first you have to choose."
Tom did not listen. He saw a car on the ground. He grabbed it and ran to his mom. His mom saw the car had a lot of fun.
"Why are you going to the jet?" Tom asked.
"Not for many things to choose," his mom said. She was worried about Tom. He tried to pick it up, but his mom said no. "It's not good, but your daughter is also very lucky. It makess sad. It's important to me. You should
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: ['Not']
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: ['makess']; grammatical: **False**; repetition 2.7%

### `tom_ball_1341`

**Prompt:** `Tom picked up the red ball and`

```
Tom picked up the red ball and threw it back to Lily.
Lily felt happy and wanted to stop smiling. She did not want to do it, but she knew it was too high. She said, "I can't fly again, please? I'm sorry that golf now. Can we go back to another park?"
Lily smiled and said, "Yes, we can go back now. I can fly again. And please be careful. Let's go back to the swings after climbing?"
They walked to the swings and the sun and the slide. They had a lot of fun and tired. They wished they had listened to Lily. They went back to the park to look for more birds to play. And they did not feel sorry anymore.
```

- **cat 1**: first name `Tom` -> **changes**; uninitiated new names: ['Lily']
- **cat 3**: 147 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `lily_key_1337`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She put it in her hand and continued her work.
Before long, Lily turned on the key and said goodbye. She was happy that she was safe with her unlockment all the hard. From that day on, Lily always made sure to always be safe and protect the open key for her to get it safe again.
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: ['Before']
- **cat 3**: 66 tokens, reached <EOS>; ends cleanly: True; invented words: ['unlockment']; grammatical: **False**; repetition 0.0%

### `lily_key_1338`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She sat on the bed in and was feeling very brave.
When the night was finished, Lily had forgotten about the key. She was so brave that she kept the key. The end.
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: none
- **cat 3**: 39 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `lily_key_1339`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She picked it up in her hand, and gave it to her friends.
Her friend, Sue, told them, "I found it!" Everyone smiled at her.
They decided to play together, but then, Amy and Tom had to look for the key. Instead, they played soccer and shared it with each other. They played together, and made a happy memory of the broken key.
```

- **cat 1**: first name `Lily` -> **changes**; uninitiated new names: ['Sue', 'Amy', 'Tom', 'Instead']
- **cat 3**: 83 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `lily_key_1340`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She wanted to open the door to see what it was.
She looked up at the key and saw a little figure. The wall had a hole in it. She called the door, and the door opened. Inside the house, there was a house. She was curious and climbed up top. She saw a garden in the kitchen. It was full of bugs!
Lily wanted to touch the doors again. She wondered what it was for her. She looked, but she could not see the key.
She asked her mom, "What is this door open? What is that!"
Her mom said, "I don't know. Come on! I will see it again."
But Lily knew she had to go to the house
```

- **cat 1**: first name `Lily` -> **recurs**; uninitiated new names: ['Inside']
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 0.0%

### `lily_key_1341`

**Prompt:** `Lily found a shiny key under the bed. She`

```
Lily found a shiny key under the bed. She picked it up and put it in a soft blanket. She put it in her mouth and ran back inside.
On her way, when she opened the lock, there was a funny poison on the floor. It was the best part she had ever been!
The moral of the story is that you can give anything you think. Don’t give it back to your friend, it is important to protect your friends. You are the tougherest reless ruinarian, even if it's important!"
```

- **cat 1**: first name `Lily` -> **disappears**; uninitiated new names: none
- **cat 3**: 104 tokens, reached <EOS>; ends cleanly: True; invented words: ['tougherest', 'reless', 'ruinarian']; grammatical: **False**; repetition 0.0%

### `rainy_friends_1337`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and had a magical adventure.
The end.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 9 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `rainy_friends_1338`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and drank the water. The bunny was feeling very brave and said, "Let's play!" The bunny hopped back inside, happy, and the bunny felt brave again.
As the bunny went exploring, they saw a big tree that was very small and could not find the sun. The bunny took the rabbit home to take him to the castle, and the bunny gave him a big hug. From that day on, the bunny and bunny all played together in the big tree every day. The end.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 100 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.1%

### `rainy_friends_1339`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and enjoyed the rain in the rain, they laughed and played and laughing. An guess, they were very happy!
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 23 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `rainy_friends_1340`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and did everything again.
Later, the rain started to melt. Tom and Baby were scared at first, so they tried to jump. Eventually, they were still stuck, but Tom and Baby were still scared. The rain was not safe again.
Tom and Baby were very sorry and said sorry to each other. They were still together, but they knew they wanted to make a new house for their own. So, Tom and Baby played with their own toys and had many things. They learned that it is important that it's always important to be safe.
```

- **cat 1**: first name `Tom` -> **recurs**; uninitiated new names: ['Baby', 'Eventually']
- **cat 3**: 116 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 5.3%

### `rainy_friends_1341`

**Prompt:** `It was raining outside, so the two friends stayed inside and`

```
It was raining outside, so the two friends stayed inside and waited for the next adventure.
From that day forward on, the friends made sure to explore and explore everywhere, when it was time for the party, they decided it was time to go to the park and play with its special treat.
The two friends laughed and hugged each other, and they went back home.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 64 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `cat_window_1337`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the cat stopped. The lady was still scared, so she ran away.
The old man hugged the little girl and said, "There is no animal in the window! You should go back to bed alone and safe. Don't scare the animals away." The little girl thanked the old owl, and waved goodbye as the old voice walked away.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 70 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `cat_window_1338`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the dog heard some noise and was feeling sorry. He said, "Are you okay, little bird?"
The old cat replied, "I told you, I was brave and brave. Let's play together." And they both agreed.
The old cat said, "Let's go play with the dog to the park. It's time for the cat." The old cat agreed the story. He had learned a valuable lesson—It’s lesson from others.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 93 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 2.2%

### `cat_window_1339`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the house started to tremble to shake and shake and shookterrautions. 
The dog and the dog ran back inside and destroyed their house. The farmer told the dog, "We must pune for your house and it won't break." 
The angry dog didn't listen and just smiled.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 66 tokens, reached <EOS>; ends cleanly: True; invented words: ['shookterrautions', 'pune']; grammatical: **False**; repetition 0.0%

### `cat_window_1340`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, a little rabbit appeared from the window. The cat said, "Hello my name is Anna. What's your name?" The white bird replied, "My name is Polly, who loves me. Don't worry about me. Go away, let's go call I'll play." 
The blue bird did not say anything, but it said, "You'll be my friend." Eventually, the brown bird flew away happily and the pink cat came back. The orange bird said, "Thank you, mister. You are a good friend." The white bird replied, "You're welcome, but let's go to our destination." So, the red bird and Polly decided to go find out where Fluffy had back. They were happy and
```

- **cat 1**: first name `Anna` -> **changes**; uninitiated new names: ['Polly', 'Eventually', 'Fluffy']
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 4.1%

### `cat_window_1341`

**Prompt:** `The old cat sat by the window watching the birds. Suddenly,`

```
The old cat sat by the window watching the birds. Suddenly, the mouse noticed that the cat was dead.
The mouse said: "Why don't you look nice and clean the cat and so I can help you." The mouse quickly ran to get to the mouse and started to pick up the mouse. The mouse was very happy and thanked the mouse.
The mouse and the mouse became friends and became good friends. And they all lived happily ever after. And they lived happily ever after.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 87 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 7.1%

### `park_dialogue_1337`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he had saved his adventure.
Ben was relieved that the park was much fixed and fixed the path. He also gave the car back to the family. He thanked her for saving all the video and they were both happy.
From that day on, Ben knew that he should always remember the day he passed with a new skull.
```

- **cat 1**: first name `Ben` -> **recurs**; uninitiated new names: none
- **cat 3**: 70 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `park_dialogue_1338`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he shouted, "Yay! You are such a great way to play in the park and build your bike golf!" Mom gave him a big hug and said, "You are very creative and creative."
Now Ben is the best girl ever has anything she likes.
```

- **cat 1**: first name `Ben` -> **recurs**; uninitiated new names: ['Yay']
- **cat 3**: 53 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `park_dialogue_1339`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he jumped up and down.
```

- **cat 1**: first name `Ben` -> **disappears**; uninitiated new names: none
- **cat 3**: 5 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `park_dialogue_1340`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he did.
Lily and Ben played all day. They forgot about the park, and they became good friends. They stayed close to the park and played together.
```

- **cat 1**: first name `Ben` -> **recurs**; uninitiated new names: ['Lily']
- **cat 3**: 32 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `park_dialogue_1341`

**Prompt:** `Mom said, "We are going to the park today!" Ben was so happy that he`

```
Mom said, "We are going to the park today!" Ben was so happy that he forgot about the park that was safe and he was very glad to be in his new friend home.
```

- **cat 1**: first name `Ben` -> **disappears**; uninitiated new names: none
- **cat 3**: 20 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `robot_wish_1337`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to find out. The robot was very sad. It asked a friendly old man who lost his toy. The man gave the little boy some money. The little boy was so happy and they played together. They became best of friends.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 46 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `robot_wish_1338`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to make the vroom in the jplipted. He was very selfish and tried to take him away. But it was too hard for him.
After he was done, the robot had a new home. He thanked the robot and ran back home. But no matter what the important robot was to do, the robot didn't always work. The end.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 74 tokens, reached <EOS>; ends cleanly: True; invented words: ['jplipted']; grammatical: **False**; repetition 0.0%

### `robot_wish_1339`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to move his head in the hole.
The robot told a hard plan to move the robot with his brand new brand new click. The robot tried to fly faster, but it was too heavy. Suddenly enough, the robot started to move. It moved over and over again, so it moved and rolled through the hole.
There, the little boy was very happy that the robot was able to move his arms, and it felt like he was flying again. The robot thanked the robot and took the robot outside. The moral of this story is to learn new things, just need to help solve other things, even if we love it.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 131 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 1.6%

### `robot_wish_1340`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to do something to do. The robot would help the robot would come to the world and be independent by himself.
One day, the robot and the robot were very sad, but he did not know how to help the robot, if he could come with them. He promised that it would make the robot stay safe in the box.
The robot had a lot of fun with the robot made the robot go very fast. He was so excited to hear what was happening and learned a lesson - that the robot was just an important thing that made a bad ending.
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 113 tokens, reached <EOS>; ends cleanly: True; invented words: none; grammatical: **True**; repetition 0.0%

### `robot_wish_1341`

**Prompt:** `Once upon a time, there was a tiny robot who wanted to`

```
Once upon a time, there was a tiny robot who wanted to play in the world. The robot would attach out a toy to everyone. One day, the robot started to feel sad. He didn't think it was wrong.
The robot decided to attach the toy to its special toy. The robot was very smart and used it to attach its body to the bird to its body. The robot tried to figure out the wire, but it just wouldn't budge.
Then, the robot started to work. The robot ran, grabbed the laser with its sharp paw. The robot was so excited to be free! The robot cheered and waved his tail.
The robot continued its train, and the robot became kind again. It took the toy and the magnet and played with the robot all
```

- **cat 1**: first name `None` -> **no name**; uninitiated new names: none
- **cat 3**: 150 tokens, hit the cap; ends cleanly: False; invented words: none; grammatical: **False**; repetition 5.4%

