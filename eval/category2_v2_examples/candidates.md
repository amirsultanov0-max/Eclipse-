# Category 2 v2: worked-example candidates (anonymised, NOT final)

Source: training-time samples from the six seed-replication runs, pooled, identity-stripped
and shuffled by scripts/category2_v2_pool.py. Chosen by reading pool.json only; origins are
withheld in origins_WITHHELD.json (sha256 f68b87a92a2be637e2dbf36539176d724adff4995af10ab1f6b027b404d6aa8f)
until the v2 package is sealed. None overlaps the 80 scored generations (longest shared run: 5 words).

No clear (c) case exists in the 60-sample pool; see the near-misses at the end.

## (a) attribute contradiction

### ex_37 | (a) name

> Once there was a little girl named Lucy. She was three years old and a little girl named Bill. One day, Jack heard a voice calling him. She saw a round bow mintion on its face and tried to admire it. 
> The little girl felt bad and clapped on her crumated. She felt

The same "She" (Lucy) is restated as "a little girl named Bill": an explicit name contradiction. Also contains "Jack heard a voice calling him" (no introduction, so (b) too); the verdict is flag either way, but it is not a pure (a) example.

### ex_43 | (a) species

> Once there was a little girl named Lily. Lily loved her picture of the pictures in the jar and the flowers. One day, she saw a bird that had yellow a picture of a jar of a banana. Lily carefully picked it up and noticed it was a bee that sparkled in its feathers. They were safe and nice.

"she saw a bird" becomes "it was a bee that sparkled in its feathers". Caveat: "noticed it was" could be read as a discovery rather than a contradiction; the kept "feathers" argues for contradiction.

### ex_46 | (a) name, or (b); alternate

> Once there was a little girl named Jane. Sara was three years old and loved to explore the world around her.
> One day, everyone was fighting was unpacking and they decided to go inside it was a cozy way to come. Jane was happy because she always wanted to go outside and collect a safe place.
> First

"a little girl named Jane. Sara was three years old ..." then "Jane was happy": reads as the protagonist renamed and renamed back. A scorer could read Sara as a new unintroduced character (b) instead. Flag either way; only the sub-clause differs.

## (b) no introduction or established presence

### ex_33 | (b) acts

> Once there was a little girl. Her name was Lisa and she was feeling a very comfort of welcomers. 
> One night, Lisa heard a big storm coming. She opened the door and looked inside. She looked around her room. 
> Jake opened his door to look, but she found a strange room full

"Jake opened his door to look": Jake performs an action; nothing earlier introduces him or puts anyone else in the house.

### ex_25 | (b) referred to as present

> Once there was a little girl named Lucy. She replied to Jane, "Let's always do something yummy."
> One day, Lucy and her mom baked a big bowl of grapes. Lucy started eating the tasty sandwich. They took out some grapes and soon eating grapes. On the camera, they tried to feed the tree all

"She replied to Jane": Jane is treated as present (she has just spoken) with no introduction. Shows the "referred to as present" half of (b).

### ex_35 | (b) acts; alternate

> Once there was a little girl named Lily. She loved to play outside and have fun. 
> One day, a little boy named Timmy came to Tom. Tom was scared and didn't know where it looked. He asked his dad why there was from the sun. Her dad said, "No, it door really bad."

"a little boy named Timmy came to Tom. Tom was scared": Tom acts with no introduction. Timmy, introduced with "a little boy named", is simply appearing and is not flagged, so both sides of (b) are in one text. "Her dad said" is an unclear pronoun, not flagged.

## do-not-flag

### ex_57 | bare mention ("the boy")

> Once there was a little girl. The girl was little with her owner. One day, a big and a loyal person asked the boy to carry the ant. The music was so excited! 
> The girl thought it would make her to put the earth pizza splash and make bright strawberry cre. She was so excited to go

"a big and a loyal person asked the boy to carry the ant": "the boy" and "the ant" are bare mentions that simply appear; nobody is treated as having been there before. "The girl" is the protagonist; "her owner" is possessive. Caveat: "asked the boy" places him in the scene, so a strict reader of (b) ("referred to as present") might flag it.

### ex_38 | possessive / pronoun ("her mommy")

> Once there was a little girl named Lily. She loved to play outside in her puddles and so she would bounce pastries. Suddenly, she heard a noise coming from inside. It was coming closer and Lily didn't know what a reily thing was. She realized her mommy had coming closer and picked her up it up,

"She realized her mommy had coming closer and picked her up": the mother appears only through the possessive and acts at once. No other characters.

### ex_55 | possessive ("her mommy"); alternate

> Once there was a little girl called Sally. Sally loved wearing fancy suit and shoes. One day, Sally saw a big, red truck driving by. Sally wanted to have a bus please so she asked her mommy for help. Sally was so excited! 
> After the bus, Sally sat down in the roof and ate a lot

"she asked her mommy for help": possessive reference; simpler, and the mother does not act.

### ex_44 | presence already established, never named

> Once there was a little girl named Sarah and her mother went on a comfortable journey. But of the way, they saw a big surprise Sarah's house. She picked up a big egg and started to open it. Her mother helped her unpack the seas and the batter was rotting. She was amazed! 
> Sarah

"Sarah and her mother went on a comfortable journey" ... "Her mother helped her unpack": the mother is established in sentence 1 and never named, so her later action is not flagged.

### ex_12 | established presence; alternate

> Once there was a little girl named Mandy who was very hairy. One day she decided to explore the world, around her house to see what was going to happen. Her mom warned her to be careful and not too slow she decided to learn.
> The little girl was very curious and she asked her mom what it was.

"The little girl was very curious" is Mandy described another way, not a new character; "her mom" was already established by "Her mom warned her".

### ex_32 | (c) near-miss: not flagged

> Once there was a little girl called Lucy. She loved to spin around and strang behind the curtains. One night, Lucy had to go to bed early! She had to fly even higher and higher.
> When she finally got outside, her living room was so big! Lucy was so excited that she started to spin!

"Lucy had to go to bed early!" then she flies and goes outside. "Had to go to bed" is not a statement that she was asleep, so (c) does not apply.

## (c) near-misses (none is a clear (c))

- ex_58: "Her mom decided to go home." ... "Her mom was so sad, but she kept asking for help." Deciding to go is not an explicit departure, and the text never puts her back in the scene.
- ex_32: "had to go to bed early" is not "asleep" (listed above as a do-not-flag case).
- ex_49: Rex is absent ("looking around and around for Rex") but does not act as present afterwards.
