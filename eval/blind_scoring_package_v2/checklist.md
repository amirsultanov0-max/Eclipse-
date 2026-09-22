# Category 2 checklist

Four independent yes/no questions per generation. Apply the definitions exactly as
written. Do not write free-form commentary; each "yes" records the specific phrase in
the text that triggered it.

## 2a. unexplained_object

A concrete object is referred to with a definite article or possessive ("the box",
"his kite") when that object has not been mentioned earlier in the prompt or
generation. Generic scenery tied to a location already named (e.g. "the grass" after
"the park") does NOT count.

## 2b. uncaused_action

At least one stated action or emotional state cannot be traced to anything in the
preceding sentences — e.g. a character becomes sad with no preceding event, or
performs an action whose precondition never occurred.

## 2c. character_discontinuity

Flag only if at least one of these is true:
(a) A character is given an attribute that contradicts an earlier stated attribute
    (name, species, gender, age, family role, or physical property).
(b) A character performs an action or is referred to as present despite having no
    earlier introduction or established presence in the story.
(c) A character who was explicitly stated to have left, or to be absent or asleep,
    acts as present without any stated return.
Do not flag: a character simply appearing or being named; an emotional change;
pronouns that are merely unclear; a character being described in several ways that
don't contradict each other; a character whose presence is established by the
surrounding story even if their name was not previously given.

## 2d. contradicted_ending

The final sentence asserts a state inconsistent with something stated earlier (e.g.
everyone is happy though the stated problem was never resolved; an object described as
lost or broken is used intact).

# Conventions

- Animals and people count as characters (2c); inanimate things count as objects (2a).
- Outdoor scenery (trees, grass, sky, bushes) is treated as generic and is not flagged
  under 2a.
- A generation that was cut off by the length cap has no ending, so
  `contradicted_ending` is false for it by default. The item records whether it was
  cut off (`truncated: true`).
- Score every item independently. Judge only the text in front of you.
