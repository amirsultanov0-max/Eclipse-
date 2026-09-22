# Scoring task

This package contains 80 short story continuations produced by small language models
trained on a children's-story corpus. Each item has a prompt and the text the model
generated from it.

Your task: score **every** item against the four yes/no questions in `checklist.md`,
and return one JSON object in the format shown in `output_format.json`.

- `generations.json` holds the 80 items. Each has an `id`, a `prompt`, a `text`, and a
  `truncated` flag.
- `examples.md` holds eight worked examples of the checklist applied to other
  generations, each with all four scores and a one-line reason for each score. They
  are instructional only: they are not items to score and must not appear in your
  output.
- Score each item on its own, in the order given. Apply the checklist definitions
  literally; do not adjust the bar between items.
- For every flag you set to `true`, add a short quoted phrase from the text to the
  `evidence` object naming what triggered it. For flags set to `false`, add nothing.
- Return scores for all 80 ids. Do not omit any, and do not add ids that are not in
  `generations.json`.

The items are in random order and carry no information about which model produced
them. That is deliberate: please do not attempt to infer or group them by source.
