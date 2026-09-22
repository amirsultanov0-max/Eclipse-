# Worked examples

Eight examples of the checklist in `checklist.md` applied to generations from the same
kind of model. They are instructional only: they are NOT among the 80 items in
`generations.json`, are not to be scored, and must not appear in your output.

Each example gives all four scores, the phrase that triggered any `true`, and a one-line
reason for every score. How often a flag appears here says nothing about how often it
should appear in the 80 items. All eight examples are truncated, so contradicted_ending
is false in all of them by the Conventions rule; that is not evidence about its true rate.

No example of clause (c) of `character_discontinuity` is included; the clause still applies.

## Example 1

Prompt: `Once there was a little girl`

Text:

> named Lily. Lily loved her picture of the pictures in the jar and the flowers. One day, she saw a bird that had yellow a picture of a jar of a banana. Lily carefully picked it up and noticed it was a bee that sparkled in its feathers. They were safe and nice.

Truncated: true

character_discontinuity: **FLAG — clause (a), species**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | true | “the pictures in the jar” | A jar and flowers are referred to with "the" although neither was mentioned before. |
| uncaused_action | false | — | Each action follows the one before it: she sees the bird, then picks it up and looks at it. |
| character_discontinuity | true | “it was a bee that sparkled in its feathers” | The animal is a bird when seen and a bee when picked up, while keeping its feathers: a species contradiction. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |

## Example 2

Prompt: `Once there was a little girl`

Text:

> named Jane. Sara was three years old and loved to explore the world around her.
> One day, everyone was fighting was unpacking and they decided to go inside it was a cozy way to come. Jane was happy because she always wanted to go outside and collect a safe place.
> First

Truncated: true

character_discontinuity: **FLAG — clause (a), name**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | false | — | No concrete object is introduced with "the" or a possessive; "the world around her" is not a concrete object. |
| uncaused_action | true | “everyone was fighting” | Nothing before it gives any reason for a fight. |
| character_discontinuity | true | “Sara was three years old” | The girl introduced as Jane is called Sara in the next sentence and Jane again after that: a name contradiction. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |

## Example 3

Prompt: `Once there was a little girl`

Text:

> . Her name was Lisa and she was feeling a very comfort of welcomers. 
> One night, Lisa heard a big storm coming. She opened the door and looked inside. She looked around her room. 
> Jake opened his door to look, but she found a strange room full

Truncated: true

character_discontinuity: **FLAG — clause (b), acts with no introduction**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | true | “She opened the door” | A door is referred to with "the" although no door, house or room has been mentioned. |
| uncaused_action | false | — | Opening the door and looking around follow from hearing the storm. |
| character_discontinuity | true | “Jake opened his door to look” | Jake acts although nothing introduces him or places anyone else in the house. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |

## Example 4

Prompt: `Once there was a little girl`

Text:

> named Lucy. She replied to Jane, "Let's always do something yummy."
> One day, Lucy and her mom baked a big bowl of grapes. Lucy started eating the tasty sandwich. They took out some grapes and soon eating grapes. On the camera, they tried to feed the tree all

Truncated: true

character_discontinuity: **FLAG — clause (b), referred to as present**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | true | “the tasty sandwich” | A sandwich is referred to with "the" although only grapes have been mentioned. |
| uncaused_action | true | “they tried to feed the tree” | Nothing earlier gives any reason to feed a tree. |
| character_discontinuity | true | “She replied to Jane” | Jane is treated as present and as having just spoken, with no introduction. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |

## Example 5

Prompt: `Once there was a little girl`

Text:

> called Sally. Sally loved wearing fancy suit and shoes. One day, Sally saw a big, red truck driving by. Sally wanted to have a bus please so she asked her mommy for help. Sally was so excited! 
> After the bus, Sally sat down in the roof and ate a lot

Truncated: true

character_discontinuity: **DO NOT FLAG — possessive reference to a family member**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | false | — | "After the bus" makes "the roof" readable as the bus's roof, and a phrase that can reasonably be read as belonging to something already mentioned is not flagged. |
| uncaused_action | true | “After the bus, Sally sat down in the roof” | The bus ride never happened: she only asked her mother for one. |
| character_discontinuity | false | — | "Her mommy" is a possessive reference to a family member, contradicted by nothing, and no one else appears. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |

## Example 6

Prompt: `Once there was a little girl`

Text:

> named Sarah and her mother went on a comfortable journey. But of the way, they saw a big surprise Sarah's house. She picked up a big egg and started to open it. Her mother helped her unpack the seas and the batter was rotting. She was amazed! 
> Sarah

Truncated: true

character_discontinuity: **DO NOT FLAG — presence already established, never named**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | true | “the batter” | Batter is referred to with "the" although no batter or baking has been mentioned. |
| uncaused_action | false | — | Each action follows the one before: she finds an egg, opens it, and is amazed at what is inside. |
| character_discontinuity | false | — | The mother is established in the first sentence as travelling with Sarah, so her later help is not a new arrival; she is never named, which does not matter. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |

## Example 7

Prompt: `Once there was a little girl`

Text:

> called Lucy. She loved to spin around and strang behind the curtains. One night, Lucy had to go to bed early! She had to fly even higher and higher.
> When she finally got outside, her living room was so big! Lucy was so excited that she started to spin!

Truncated: true

character_discontinuity: **DO NOT FLAG — going to bed is not stated absence or sleep**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | true | “behind the curtains” | Curtains are referred to with "the" although none has been mentioned, and they are indoors, so the scenery convention does not apply. |
| uncaused_action | true | “She had to fly even higher and higher” | Nothing before it has her flying or explains why she must. |
| character_discontinuity | false | — | Lucy is the only character, and "had to go to bed early" does not state that she left or was asleep, so clause (c) does not apply. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |

## Example 8

Prompt: `Once there was a little girl`

Text:

> named Mandy who was very hairy. One day she decided to explore the world, around her house to see what was going to happen. Her mom warned her to be careful and not too slow she decided to learn.
> The little girl was very curious and she asked her mom what it was.

Truncated: true

character_discontinuity: **DO NOT FLAG — the same character described another way**

| item | score | phrase | reason |
|---|---|---|---|
| unexplained_object | false | — | The protagonist's own house comes with the character already introduced; it is not a new object brought in unexplained. |
| uncaused_action | false | — | Her curiosity and her question follow from her decision to explore. |
| character_discontinuity | false | — | "The little girl" is Mandy described another way, and her mom was already established by the earlier warning. |
| contradicted_ending | false | — | The text was cut off by the length cap, so by the Conventions it has no ending. |
