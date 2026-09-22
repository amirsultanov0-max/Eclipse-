# Stage 6.2 option 1 — equivalence evidence

Artifacts behind the equivalence claim in `eval/stage6_2_preregistration.md` sections 9, 10 and 13:
that adding `--d-model`, `--n-heads` and `--d-ff` to `train_transformer.py` leaves a default
invocation semantically identical to the code the reference arm was trained under.

Pre-change code: `train_transformer.py` sha256 `817e03ec…` (content at `c4db503`).
Post-change code: sha256 `57d4e57e…`. `model/transformer.py` is unchanged (`63ad52cd…`).

## Why the bitwise comparison ran on CPU

MPS is not bit-reproducible run to run. Control: rerunning the recorded `sr_data200mb_s1337` run's
first 1,001 steps with unchanged code and the same seed reproduced the batch start positions for
all 1,001 steps but diverged in loss from step 5 and in gradient norm from step 7. A bitwise
comparison of losses or weights on MPS is therefore impossible regardless of any code change.

CPU arithmetic is deterministic here: two 20-step CPU runs of identical code produced bitwise-equal
per-step logs and identical initial, final and eval fingerprints. The bitwise comparison was run on
CPU for that reason, and the MPS-only quantities that ARE deterministic were compared separately on
MPS.

## Results

1,001 steps, default arguments, seed 1337, 200 MB (6.1) training stream.

| check | device | result |
|---|---|---|
| per-step loss, all 1,001 steps | CPU | bitwise equal |
| per-step gradient norm, all 1,001 steps | CPU | bitwise equal |
| per-step batch start positions, all 1,001 steps | CPU | bitwise equal |
| whole per-step log file | CPU | bitwise equal |
| initial weights fingerprint | CPU | bitwise equal |
| eval-batch fingerprint | CPU | bitwise equal |
| final weights fingerprint | CPU | bitwise equal |
| step-1,000 generation sample text | CPU | bitwise equal |
| step-1,000 generation draw count | CPU | equal (60) |
| initial weights fingerprint | MPS | bitwise equal |
| eval-batch fingerprint | MPS | bitwise equal |
| running batch-start hash over 1,001 steps | MPS | bitwise equal |
| per-step loss | MPS | differs at 906 of 1,001 steps, largest gap 3.3e-03 — the machine's own run-to-run noise, not the edit |

## Generation does not touch the training-batch stream

The batch start positions were reproduced independently of the training script (seed, construct the
model so its init draws are consumed in the same order, then draw 1,001 batches). All 1,001 hashes
matched on both CPU and MPS, including step 1,001, the step immediately after the step-1,000
in-training generation. Generation therefore drew only from its own generator.

## Amendment 2: eval reads the architecture from the checkpoint config

`eval_official_valid.py` pre-change sha256 `06293fa4…`, post-change `ccc7bb9b…`. Both official
evaluations were rerun on `sr_data200mb_s1337_best.pt` with the post-change code and reproduced the
committed values exactly: full file 2.1065 (cross-check 2.1065 over 4,796,928 tokens), clean subset
2.1080 (cross-check 2.1079 over 4,766,208 tokens). The regenerated reports
(`regression_official_full.md`, `regression_official_clean.md`) are byte-identical to the committed
ones apart from their timestamp line.

## Files

| file | contents |
|---|---|
| `eq_pre_cpu_grad.csv`, `eq_post_cpu_grad.csv` | per-step loss, gradient norm and batch-start hash, CPU, pre and post |
| `eq_pre_grad.csv`, `eq_post_mps_grad.csv` | the same on MPS |
| `eq_pre_cpu_samples.txt`, `eq_post_cpu_samples.txt` | the step-1,000 generation, CPU, pre and post |
| `eq_*.provenance.json` | run manifests with the weight, eval-batch and batch-start fingerprints |
| `smoke_d352.provenance.json` | the 50-step d352 smoke test: 10,537,472 parameters, d_ff 1408, 4 heads |
| `regression_official_full.md`, `regression_official_clean.md` | amendment 2 regression: both official evals rerun on the reference-arm seed-1337 checkpoint |
