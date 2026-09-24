# Eclipse

Eclipse is a small decoder-only Transformer language model, written and trained from
scratch in PyTorch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
a dataset of short, simple children's stories. You give it the beginning of a story and it
continues it. It runs locally in a browser page or from the command line.

Eclipse is a **text-completion model, not an assistant**. It was not trained on
instructions or dialogue, so it does not answer questions or follow requests. It writes
in the style of stories for young children and nothing else.

The project had three phases:

1. **Building the system**: tokenizer, Transformer, training pipeline, checkpointing.
2. **Controlled experiments** on data scale and model capacity at a fixed training budget.
   The final write-up is [`eval/track_a_conclusion.md`](eval/track_a_conclusion.md).
3. **The program in this README**: loading the trained model and serving it to a person.

Phase 3 adds code around the model without changing it. The model, tokenizer and
Phase 2 research record are used as they are.

## The model Eclipse serves

| | |
|---|---|
| Checkpoint | `sr_data200mb_s1337_best.pt`, the 200 MB-data, seed-1337 reference run from the seed-replication study |
| Architecture | 6 Pre-LN blocks, d_model 128, 4 heads, d_ff 512, learned positions, tied embeddings |
| Parameters | 1,767,424 |
| Context | 512 tokens |
| Tokenizer | from-scratch BPE with 4,000 tokens, `tokenizer/bpe_4000.json` |
| Training | 10,000 steps, batch 16, 81.9M token positions, AdamW lr 1e-3 constant |
| Validation loss | 2.0860 nats on the monitoring split, 2.1065 on the official TinyStories validation file |

The larger d_model 352 model from Stage 6.2 has a lower loss, but it was a research
comparison arm, not a deployment candidate, so Eclipse does not serve it.

## Setup

You need Python 3.11 or newer.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.14.0`, the version every Phase 2 run was trained with.
On Linux without a GPU you can install the smaller CPU-only build first:
`venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`.

### The checkpoint

The trained weights are not in this repository: `checkpoints/` is gitignored. Copy the
file to this location:

```
checkpoints/sr_data200mb_s1337_best.pt
```

Training originally wrote it to
`checkpoints/seed_replication/sr_data200mb_s1337/sr_data200mb_s1337_best.pt`. To use a
different location, pass `--checkpoint PATH` or set `ECLIPSE_CHECKPOINT=PATH`.

When Eclipse loads the checkpoint it runs these checks, and it refuses to start if any
of them fails:

- **The file must exist.** If it is missing, Eclipse stops with an error. It never falls
  back to another model.
- **The file is opened safely**, with `torch.load(weights_only=True)`, so it cannot run
  code while loading.
- **The architecture comes from the checkpoint.** Nothing is hardcoded. Values recorded
  in the checkpoint's `config` are checked against the shapes of its weights. This
  checkpoint was saved before the training script recorded `d_model`, `d_ff` and
  `n_heads`, so d_model, d_ff and the block count are read from the weight shapes. The
  head count leaves no trace in the weights. Every checkpoint from that period was built
  with the model's default of 4 heads, so Eclipse uses 4 and reports that it did.
- **The tokenizer is the frozen research tokenizer.** Its SHA256 must match the value
  recorded in `eval/stage6_2_preregistration.md`, and its vocabulary size must match the
  model's.
- **The weights are the recorded training run's weights.** The checkpoint's config names
  its provenance manifest, `results/seed_replication/sr_data200mb_s1337.provenance.json`.
  The best checkpoint is the final step, 10,000, so Eclipse hashes the weights the same
  way training did and compares the result with the manifest's `final_weights_sha256`.
  If they differ, Eclipse refuses to load. `--skip-provenance-check` overrides this and
  serves the file anyway, and `/api/info` then reports `"skipped"`.

## Running Eclipse

### In the browser

```bash
venv/bin/python server.py
```

Then open <http://127.0.0.1:8000>. The model is loaded once, before the server starts
accepting requests. By default the server listens only on this machine. Use
`--host 0.0.0.0` to expose it on your network, and `--port` to change the port.

On the page:

1. Type the beginning of a story, or click one of the examples.
2. Optionally adjust the generation settings.
3. Click **Continue the story**, or press Ctrl/⌘ + Enter.

Your text is shown in grey and Eclipse's continuation is highlighted. Below it the page
shows how many tokens were generated, why generation stopped, and the seed that was
used. The footer shows which checkpoint is loaded, its architecture, and the result of
the provenance check.

### From the command line

```bash
venv/bin/python generate.py "Once upon a time, there was a tiny robot who wanted to"
venv/bin/python generate.py "Lily found a shiny key under the bed. She" --temperature 0
venv/bin/python generate.py "Tom picked up the red ball and" --seed 1337 --max-new-tokens 200
venv/bin/python generate.py --info        # what was loaded, as JSON
```

The story is printed to stdout. The details (token count, stop reason and seed) go to
stderr.

### How Eclipse works

<http://127.0.0.1:8000/model>, linked from the main page, has two parts:

1. **Architecture.** A block diagram of the loaded model: token and position
   embeddings, the stack of Transformer blocks (attention, then feed-forward, each with a
   residual connection), the final LayerNorm, and the output head, which reuses the token
   embedding matrix. Every dimension and parameter count comes from the loaded
   checkpoint's weights.
2. **Attention.** Type a short phrase (at most 30 tokens) and pick a block. For each of
   that block's heads, a heatmap shows how much each token attends to each earlier token.
   Rows are the tokens doing the looking and each row sums to 1. The upper triangle is
   blank because a token cannot see later tokens. Hover over a cell for its exact
   weight, or open the table view.

The attention weights are real: they come from one ordinary forward pass of the loaded
model on that exact phrase. `model/transformer.py` is frozen by the Stage 6.2
preregistration, so it is not modified to expose them. Instead, `eclipse/attention.py`
attaches PyTorch forward hooks to each block's Q, K and V projections and to the input
of its output projection, and removes them afterwards. The model keeps its weights in a
local variable that hooks cannot reach, so they are recomputed from the captured Q and K
with the same operations `forward()` uses.

Every request is verified before anything is returned: the weights multiplied by V must
reproduce, bit for bit, the attention output the model itself produced in that pass. If
any block fails, the server returns an error instead of weights. The tests also check
the weights bit for bit against the softmax output computed inside `forward()`.

### Eclipse in 3D

<http://127.0.0.1:8000/model-3d> has two rotatable 3D views. It is not linked from the
other pages yet, so open it by its URL. Drag to rotate, scroll or pinch to zoom, and
hover over (or tap) a point to read it.

1. **Embedding sphere.** Each point is a token, placed by the first three principal
   components of the trained 4,000 × 128 token embedding matrix, computed with a plain
   numpy SVD. Three components keep only part of that matrix's variance, and the page
   always shows how much. By default every point sits on a sphere at the direction of its
   three components. That direction is real, but distance from the centre is not.
   **Raw PCA** shows the actual coordinates instead. Lines join each token to its most
   similar tokens among those shown. Similarity is cosine similarity across all 128
   dimensions, not distance in the picture, and brighter lines are more similar.
   Hovering a token shows its nearest neighbours in the whole 4,000-token vocabulary.
   The page shows 300 tokens by default (100 on phones): the first ones the tokenizer
   learned that are a space followed by at least three letters.
2. **Attention sphere.** The same verified attention weights as the flat page, from
   the same `/api/attention` endpoint, drawn as a ring of tokens. Each line runs from a
   token to an earlier token it attends to, with brightness and particle speed set by the
   weight. There are never lines to later tokens. A slider hides weights below a
   threshold, and the page reports how many it hides.

Every point, line and number comes from the loaded checkpoint. Colours, glow and motion
are presentation only. The page uses [Three.js](https://threejs.org) 0.185.0 (MIT
licence), vendored in `web/vendor/three-0.185.0/` and served locally like every other
file, so the app still loads nothing from other hosts. The files are the ones in the
npm release tarball, whose sha512 matched the registry's published integrity value, and
the tests pin their SHA256 hashes.

### HTTP API

`POST /api/generate`

```json
{"prompt": "Once upon a time", "max_new_tokens": 150, "temperature": 1.0, "top_k": 40, "seed": null}
```

Only `prompt` is required. An example response:

```json
{"prompt": "Once upon a time", "completion": ", there was a little girl named...",
 "prompt_tokens": 4, "prompt_unknown_tokens": 0, "generated_tokens": 87,
 "stop_reason": "eos", "seed": 2817340551,
 "settings": {"max_new_tokens": 150, "temperature": 1.0, "top_k": 40, "seed": null},
 "elapsed_ms": 412.3}
```

An invalid request gets HTTP 422 with an explanation in `{"error": "..."}`. This covers
a missing, empty or overlong prompt, out-of-range settings, wrong types and unknown
fields.

`GET /api/info` returns the loaded checkpoint, its architecture and where each value came
from, the training configuration, the tokenizer hash, the provenance result, the device,
and the limits and defaults.

`GET /api/architecture` returns the loaded model's dimensions and its parameter count per
component: embeddings, each block's LayerNorms, attention and feed-forward layers, and
the final LayerNorm.

`POST /api/attention` takes `{"prompt": "..."}` (at most 30 tokens and 400 characters)
and returns the prompt's tokens and `attention[block][head][query][key]`, rounded to 6
decimal places, with `"verified": true`. Invalid input gets HTTP 422. A failed
verification gets HTTP 500 and no weights.

`GET /api/embedding-projection?count=300&k=3` returns `count` tokens (10–800), each with
its three PCA coordinates and its `k` (1–8) most cosine-similar tokens among those
returned, as `[token_id, similarity]` pairs. It also returns the explained-variance
ratio of the three components. The default response is about 37 KB.

`GET /api/embedding-neighbors?token_id=242&k=10` returns one token's `k` (1–20) most
cosine-similar tokens across the whole vocabulary. Out-of-range values get HTTP 422 on
both endpoints.

## Generation settings

| Setting | Default | Range | What it does |
|---|---|---|---|
| `max_new_tokens` | 150 | 1–512 | Maximum tokens to generate. Generation usually ends before this, when the model produces its end-of-story token. |
| `temperature` | 1.0 | 0–2 | Below 1 makes output more predictable, above 1 more random. **0 is greedy decoding**: always the most likely token, so the output is deterministic. |
| `top_k` | 40 | 1–4,000, or none | Sample only from the k most likely tokens. `null` in the API, empty in the browser or `--top-k 0` on the CLI turns it off. |
| `seed` | random | 0 – 2³²−1 | Makes a sampled continuation reproducible. The seed is always reported, including a random one. |

The defaults follow the Phase 2 evaluations. The 150-token cap is the one every Stage 5
generation evaluation used, and temperature 1.0 with top-k 40 is the decoding used in the
Stage 5 error analysis. [`eval/stage5_sampling_eval.md`](eval/stage5_sampling_eval.md)
compares greedy decoding with temperature, top-k and top-p sampling on an earlier
checkpoint of the same architecture. Greedy decoding looped badly, with 29.7% mean 4-gram
repetition. Top-p 0.9 did no better than top-k 40 (2.7% against 1.5%), so it is not
offered.

Other limits: a prompt can be at most 4,000 characters and at most 512 tokens, the
model's full context. Once prompt plus continuation grows past 512 tokens, the model sees
only the most recent 512, the same sliding window every generation loop in the research
code uses.

## Hardware

Eclipse runs on the CPU by default, and any recent laptop is fast enough. The weights
are about 7 MB. Most of the memory goes to PyTorch itself: the whole process used about
0.8 GB with the standard Linux PyTorch build. On a 4-core cloud CPU, a 150-token
continuation took 0.9 s and the 512-token maximum took 5.8 s. `--device mps` (Apple
Silicon) and `--device auto` are available but not needed at this size. The model was
trained on an Apple M4 MacBook Air.

## Tests

```bash
venv/bin/python -m pytest tests
```

- `tests/test_inference.py` covers checkpoint loading, architecture resolution,
  provenance and tokenizer checks, sampling, generation limits and the CLI. It also
  checks that no external model, LLM client or network client is referenced.
- `tests/test_api.py` covers the HTTP API, the pages, invalid input, and startup with and
  without a checkpoint.
- `tests/test_attention.py` covers the architecture and attention views. It checks
  that `model/transformer.py` still has its preregistered hash, and that the hooked
  weights equal, bit for bit, the softmax output inside `forward()`. It checks that
  hooks are always removed and never change the model's output, that a failed
  verification returns no weights, and it covers the limits, the endpoints and the page.
- `tests/test_embedding.py` covers the 3D page's data. It checks the PCA against an
  eigendecomposition of the covariance, and every similarity against cosine similarity
  computed separately from the `state_dict`, including brute-force top-k neighbours. It
  also covers the token-selection rule, payload sizes, the endpoints and their limits,
  the page, and the pinned SHA256 of each vendored Three.js file.

The tests do not need the trained checkpoint. They write stand-in checkpoints with the
real model class and the training script's own `save_checkpoint`, in the exact format the
real checkpoint uses. Five tests run against the real checkpoint and are skipped, with
the reason printed, until `checkpoints/sr_data200mb_s1337_best.pt` is present.

## Limitations

- **Story style only.** Eclipse learned from TinyStories and nothing else. It knows
  simple vocabulary and the shape of a children's story. Other kinds of text produce
  TinyStories-like output or nonsense.
- **Not coherent over length.** Characters change names, objects appear and disappear,
  and endings can contradict the setup. The Phase 2 evaluations document this, and a
  lower loss does not guarantee better stories.
- **Short effective memory.** The context is 512 tokens, but Phase 2 found the model
  makes little use of anything beyond about 128–256 tokens.
- **Limited character set.** The tokenizer only knows characters seen in training.
  Anything else, such as emoji or most non-Latin scripts, becomes `<UNK>`, and the
  response reports how many there were. A trailing space at the end of a prompt splits
  off as its own token, which the model rarely saw, so end the prompt on a word or
  punctuation mark.
- **Not trained to convergence**, and trained with one fixed recipe. See section 9 of
  the Track A conclusion.
- **No safety filtering.** There is no content filter. The training data is children's
  stories, but sampled text is not guaranteed to be appropriate.

## Repository layout

| Path | Phase | Role |
|---|---|---|
| `eclipse/` | 3 | inference (checkpoint loading, checks, generation), the architecture and attention views, and the embedding projection |
| `server.py`, `web/` | 3 | web server, browser interface, the "How Eclipse works" page, and the 3D page |
| `web/vendor/three-0.185.0/` | 3 | Three.js 0.185.0 (MIT), vendored for the 3D page |
| `generate.py` | 3 | command-line generation |
| `tests/` | 3 | tests for the Phase 3 program |
| `model/transformer.py` | 1 | the Transformer (used unchanged by Phase 3) |
| `tokenizer/` | 1 | the BPE tokenizer and its frozen vocabulary |
| `train_transformer.py`, `prepare_data*.py`, `lr_sweep.py` | 1–2 | training and data pipeline |
| `eval_*.py`, `eval/`, `results/`, `scripts/`, `docs/`, `metadata/` | 2 | experiments, evaluations and the research record |

Training and inference are kept separate. The Phase 3 program imports only the model
class and the tokenizer. It never imports the training or evaluation scripts, whose
hashes are frozen in the Phase 2 preregistration.
