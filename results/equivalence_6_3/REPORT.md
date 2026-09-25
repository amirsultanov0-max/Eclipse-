# Stage 6.3 Amendment 1 — equivalence evidence

Checks V1–V5 of `eval/stage6_3_preregistration.md` section 10, produced by `scripts/equivalence_6_3.py` on 2026-09-25 12:36.

**Result: ALL CHECKS PASSED.**

- post-change code: HEAD `dfe08f1badaee5bda35b05c4cad1ea6d6d775da0`, `train_transformer.py` `0acec486c1c6e197e63b72d32928dc8ca37740f30db4552f4612eb1f8f8ed762`
- pre-change code: `b0fd350` (git worktree), `train_transformer.py` `57d4e57eb7d6e6bff6c8042e2842109b0cd2f72949ef0b3ee21548842fe393a5`
- `model/transformer.py` `63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0`
- training stream `5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb`, monitoring stream `9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e`
- 1001 steps, seed 1337; environment: {"python": "3.14.7", "torch": "2.14.0", "platform": "macOS-26.6.2-arm64-arm-64bit-Mach-O", "mps_available": true}

| check | item | result | detail |
|---|---|---|---|
| V1 | per-step log has 1001 steps in both | PASS |  |
| V1 | per-step loss bitwise equal at all steps | PASS |  |
| V1 | per-step grad_norm bitwise equal at all steps | PASS |  |
| V1 | per-step batch_starts_sha256 bitwise equal at all steps | PASS |  |
| V1 | whole per-step log file bitwise equal | PASS |  |
| V1 | initial_weights_sha256 bitwise equal | PASS | a389751c22aad283… |
| V1 | eval_batches_sha256 bitwise equal | PASS | 0b16430f6461735c… |
| V1 | final_weights_sha256 bitwise equal | PASS | 64fcbbdb5b2d4e88… |
| V1 | batch_starts_running_sha256 bitwise equal | PASS | 3e667d8f8ea4f7c1… |
| V1 | in-training sample text bitwise equal | PASS |  |
| V1 | generation draw counts equal | PASS | {"1000": 60} |
| V1 | both runs read byte-identical data (manifest SHA256s) | PASS | 5c25b49be87093ce… |
| V2 | summaries identical apart from run name, date, log names, wall clock | PASS |  |
| V2 | '- model:' line unchanged | PASS | - model: 512-token context, d_model 128, 4 heads, d_ff 512, 6 Pre-LN blocks, tied LM head  |
| V2 | '- config:' line unchanged | PASS | - config: 1,001 steps this run (through step 1,001), batch 16, lr 0.001, AdamW (weight dec |
| V3 | initial_weights_sha256 bitwise equal | PASS | a389751c22aad283… |
| V3 | eval_batches_sha256 bitwise equal | PASS | 0b16430f6461735c… |
| V3 | batch_starts_running_sha256 bitwise equal | PASS | 816a41a861fe9c84… |
| V3 | per-step batch starts equal at all steps | PASS |  |
| V4 | per-step batch starts equal at all steps | PASS |  |
| V4 | initial_weights_sha256 bitwise equal | PASS | a389751c22aad283… |
| V4 | eval_batches_sha256 bitwise equal | PASS | 0b16430f6461735c… |
| V4 | batch_starts_running_sha256 bitwise equal | PASS | 3e667d8f8ea4f7c1… |
| V4 | step-1 loss equal | PASS | 8.324520111083984 |
| V4 | step-1 gradient norm equal | PASS | 2.055480718612671 |
| V4 | per-step LR file equals section 2.1 at all 1001 steps | PASS | 1001 rows, 0 mismatches |
| V5 | warmup_cosine run states its schedule | PASS | - config: 20 steps this run (through step 20), batch 16, lr 0.001 peak (linear warm-up over steps 1-1,000, then cosine d |
| V5 | d352 run states its architecture and 10,537,472 parameters | PASS | - model: 512-token context, d_model 352, 4 heads, d_ff 1408, 6 Pre-LN blocks, tied LM head — 10,537,472 parameters |
| V5 | warmup_cosine with --resume is refused before anything runs | PASS | --lr-schedule warmup_cosine cannot be combined with --resume: treatment runs are single passes from  |
| V5 | warmup_cosine with --steps 10001 is refused before anything runs | PASS | --lr-schedule warmup_cosine is defined for steps 1-10,000 only; got --steps 10,001 |

## Runs

- `pre_cpu`: `train_transformer.py --steps 1001 --seed 1337 --device cpu --train-tokens data/stage6_1_train_tokens.npy --run-name eq63_20260925120839_pre_cpu --log-name eq63_20260925120839_pre_cpu_training.csv --grad-name eq63_20260925120839_pre_cpu_grad.csv --samples-name eq63_20260925120839_pre_cpu_samples.txt --checkpoint-dir /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/ckpt_pre_cpu --results-file /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/pre_cpu.md --results-title Stage 6.3 equivalence run`
- `post_cpu`: `train_transformer.py --steps 1001 --seed 1337 --device cpu --train-tokens data/stage6_1_train_tokens.npy --run-name eq63_20260925120839_post_cpu --log-name eq63_20260925120839_post_cpu_training.csv --grad-name eq63_20260925120839_post_cpu_grad.csv --samples-name eq63_20260925120839_post_cpu_samples.txt --checkpoint-dir /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/ckpt_post_cpu --results-file /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/post_cpu.md --results-title Stage 6.3 equivalence run`
- `pre_mps`: `train_transformer.py --steps 1001 --seed 1337 --device mps --train-tokens data/stage6_1_train_tokens.npy --run-name eq63_20260925120839_pre_mps --log-name eq63_20260925120839_pre_mps_training.csv --grad-name eq63_20260925120839_pre_mps_grad.csv --samples-name eq63_20260925120839_pre_mps_samples.txt --checkpoint-dir /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/ckpt_pre_mps --results-file /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/pre_mps.md --results-title Stage 6.3 equivalence run`
- `post_mps`: `train_transformer.py --steps 1001 --seed 1337 --device mps --train-tokens data/stage6_1_train_tokens.npy --run-name eq63_20260925120839_post_mps --log-name eq63_20260925120839_post_mps_training.csv --grad-name eq63_20260925120839_post_mps_grad.csv --samples-name eq63_20260925120839_post_mps_samples.txt --checkpoint-dir /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/ckpt_post_mps --results-file /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/post_mps.md --results-title Stage 6.3 equivalence run`
- `treat_cpu`: `train_transformer.py --steps 1001 --seed 1337 --device cpu --train-tokens data/stage6_1_train_tokens.npy --run-name eq63_20260925120839_treat_cpu --log-name eq63_20260925120839_treat_cpu_training.csv --grad-name eq63_20260925120839_treat_cpu_grad.csv --samples-name eq63_20260925120839_treat_cpu_samples.txt --checkpoint-dir /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/ckpt_treat_cpu --results-file /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/treat_cpu.md --results-title Stage 6.3 equivalence run --lr-schedule warmup_cosine`
- `v5_cosine`: `train_transformer.py --steps 20 --seed 1337 --device cpu --train-tokens data/stage6_1_train_tokens.npy --run-name eq63_20260925120839_v5_cosine --log-name eq63_20260925120839_v5_cosine_training.csv --grad-name eq63_20260925120839_v5_cosine_grad.csv --samples-name eq63_20260925120839_v5_cosine_samples.txt --checkpoint-dir /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/ckpt_v5_cosine --results-file /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/v5_cosine.md --results-title Stage 6.3 equivalence run --lr-schedule warmup_cosine`
- `v5_d352`: `train_transformer.py --steps 20 --seed 1337 --device cpu --train-tokens data/stage6_1_train_tokens.npy --run-name eq63_20260925120839_v5_d352 --log-name eq63_20260925120839_v5_d352_training.csv --grad-name eq63_20260925120839_v5_d352_grad.csv --samples-name eq63_20260925120839_v5_d352_samples.txt --checkpoint-dir /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/ckpt_v5_d352 --results-file /var/folders/yb/59gfz0j11cgbkk2kltqm9gyr0000gn/T/eq63_eawt6qpp/v5_d352.md --results-title Stage 6.3 equivalence run --d-model 352 --n-heads 4 --d-ff 1408`

Files: `eq63_<run>_grad.csv` (per-step loss, gradient norm, batch-start hash), `_samples.txt`, `_lr.csv` (treatment only), `.md` (run summary), `.provenance.json`.
