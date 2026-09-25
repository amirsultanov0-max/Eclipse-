#!/usr/bin/env bash
#
# Stage 6.3 treatment arm: three 10,000-step warmup_cosine runs, strictly one at a time.
#
#   st63_cos_s1337  st63_cos_s1338  st63_cos_s1339
#   d_model 128, 4 heads, d_ff 512, 6 blocks, 1,767,424 parameters (default architecture)
#   --lr-schedule warmup_cosine: linear warm-up to 1e-3 over steps 1-1,000, cosine decay
#   to 1e-4 at step 10,000 (eval/stage6_3_preregistration.md section 2.1)
#   200 MB stream data/stage6_1_train_tokens.npy, batch 16, context 512, eval seed 1337
#
# THE BASELINE ARM IS NOT RETRAINED HERE. It is the three existing seed-replication runs
# sr_data200mb_s1337 / s1338 / s1339, trained at 0d55208 / c970064 / f0d5dbe and committed
# in 124f12f / a390499 / b5cc72f, reused under preregistration section 3. This script
# checks that they are present and unmodified, then trains only the treatment arm.
#
# The treatment invocation is the baseline invocation plus --lr-schedule warmup_cosine and
# nothing else: no --lr (default 1e-3) and no architecture flags, as the baseline runs.
#
# Per run:
#   skip if its 4 result files are already committed
#   -> tree clean -> no leftovers from an earlier attempt
#   -> section 9 provenance: train_transformer.py must hash to the Amendment 1 value
#      0acec486..., plus the model, eval, data, tokenizer and clean-subset hashes
#      (STOP before training if any fails)
#   -> train -> full-file eval -> clean-subset eval            (each under caffeinate -i)
#   -> the 4 tracked files exist
#   -> manifest checks: completed, clean tree, seed, eval seed, final step 10,000,
#      default architecture and exactly 1,767,424 parameters, the warmup_cosine schedule
#      as registered, no resume, data SHA256s, and the per-step LR log (10,000 rows,
#      ending at 1e-4)
#   -> git add exactly those 4 files -> commit "Stage 6.3: <run>"
# Any failure stops the script at once and commits nothing for that run.
# Re-running after a stop resumes at the first run that is not committed.
#
# Estimated runtime: the baseline runs took 39.3, 39.6 and 116.5 min on this machine
# (the last an unexplained outlier), so roughly 40 min per run plus ~4 min of official
# evaluation: about 2.2 h for all three if nothing throttles.
#
# Usage:
#   scripts/run_stage6_3.sh             real run
#   scripts/run_stage6_3.sh --dry-run   print every command and check, run nothing
#
# Dry-run-only test hooks, for exercising the control flow:
#   SIM_DONE="runA runB"   treat these runs as already committed
#   SIM_FAIL="run:step"    make that step fail (step = the label printed in brackets)
#
# Written for the macOS system bash (3.2).

set -u -o pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi
SIM_DONE="${SIM_DONE:-}"
SIM_FAIL="${SIM_FAIL:-}"
if (( ! DRY_RUN )) && [[ -n "$SIM_DONE$SIM_FAIL" ]]; then
    echo "SIM_DONE / SIM_FAIL are only allowed with --dry-run" >&2
    exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

PY=venv/bin/python
OUT_LOGS=logs/stage6_3
RESULTS=results/stage6_3
if (( DRY_RUN )); then
    DRIVER_LOG=$OUT_LOGS/driver.dryrun.log     # keeps dry runs out of the real log
else
    DRIVER_LOG=$OUT_LOGS/driver.log
fi
mkdir -p "$OUT_LOGS"

# Preregistration section 9. train_transformer.py is the Amendment 1 file (the treatment
# arm's code, verified in results/equivalence_6_3/); the baseline arm's own code
# (817e03ec...) is not re-verified here, because it is not retrained.
CODE_TRAIN_SHA=0acec486c1c6e197e63b72d32928dc8ca37740f30db4552f4612eb1f8f8ed762
CODE_MODEL_SHA=63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0
CODE_EVAL_SHA=ccc7bb9bd0b98aa2d2349e58329188db17d11e3a47686d77ad9c58edd187310a
DATA_TRAIN_SHA=5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb
VAL_SHA=9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e
TOKENIZER_SHA=e3876f558896533d13b7818282c6b19b55cbf205390ad4ee9e982d105762d3df
CLEAN_META_SHA=93ea3dbd08f5b421c5b3c1e9987cb891cf2d138d6ed996644b77247e9dc22d55

TRAIN_TOKENS=data/stage6_1_train_tokens.npy
VAL_TOKENS=data/stage3_val_tokens.npy
TOKENIZER=tokenizer/bpe_4000.json
CLEAN_META=metadata/stage6_1_corpus.json

SEEDS="1337 1338 1339"
STEPS=10000
SCHEDULE=warmup_cosine
PARAMS=1767424
# The reused baseline arm: seed:training-commit, from each run's provenance manifest.
BASELINE="1337:0d55208 1338:c970064 1339:f0d5dbe"

RUN=""
STEP="setup"
STAGED=0
FILES=()

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$DRIVER_LOG"
}

fail() {
    log "FAILED: run ${RUN:-<none>}, step [$STEP]. Stopping; nothing committed for this run."
    if (( STAGED )) && (( ! DRY_RUN )); then
        git reset -q -- "${FILES[@]}"
        log "unstaged: ${FILES[*]}"
    fi
    exit 1
}

# step LABEL COMMAND...
# Logs the command. In a dry run, prints only (or simulates a failure via
# SIM_FAIL). Otherwise runs it with all output teed into the driver log, and
# stops on any non-zero exit.
step() {
    STEP=$1
    shift
    log "[$RUN] [$STEP] $*"
    if (( DRY_RUN )); then
        if [[ "$SIM_FAIL" == "$RUN:$STEP" ]]; then
            log "[$RUN] [$STEP] (simulated failure)"
            fail
        fi
        return 0
    fi
    "$@" 2>&1 | tee -a "$DRIVER_LOG" || fail
}

check_clean() {
    local status
    status=$(git status --porcelain)
    if [[ -n "$status" ]]; then
        echo "working tree is not clean:"
        echo "$status"
        return 1
    fi
}

check_no_leftovers() {
    local found=0 path
    for path in "$CKPT_DIR" "$RUN_LOGS" "${FILES[@]}" "$DIFF"; do
        if [[ -e "$path" ]]; then
            echo "exists from an earlier attempt: $path"
            found=1
        fi
    done
    if (( found )); then
        echo "inspect and remove these by hand before resuming; this script never deletes."
    fi
    return $found
}

# The reused baseline arm must be present, committed and unmodified: its manifests, summaries
# and official evaluations are what the verdict compares against.
check_baseline() {
    local bad=0 pair seed commit base file
    for pair in $BASELINE; do
        seed=${pair%%:*}
        commit=${pair##*:}
        base=results/seed_replication/sr_data200mb_s$seed
        for file in "$base.md" "$base.provenance.json" "${base}_official_full.md" "${base}_official_clean.md"; do
            if ! git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
                echo "  FAIL $file is not committed"
                bad=1
            elif ! git diff --quiet HEAD -- "$file"; then
                echo "  FAIL $file differs from its committed version"
                bad=1
            fi
        done
        "$PY" - "$base.provenance.json" "$seed" "$commit" "$DATA_TRAIN_SHA" "$VAL_SHA" <<'PY' || bad=1
import json, sys
path, seed, commit, train_sha, val_sha = sys.argv[1:6]
m = json.load(open(path))
checks = {
    f"trained at {commit}": m["git"]["commit"].startswith(commit),
    "launched from a clean tree": m["git"]["dirty"] is False,
    "completed, 10,000 steps": m["status"] == "completed" and m["fingerprints"]["final_step"] == 10000,
    f"seed {seed}, eval seed 1337": m["seeds"]["seed"] == int(seed) and m["seeds"]["eval_seed"] == 1337,
    "same training and monitoring streams": (m["data"]["train_tokens"]["sha256"] == train_sha
                                             and m["data"]["val_tokens"]["sha256"] == val_sha),
    "constant LR (no schedule flag existed)": "lr_schedule" not in m["args"] and m["args"]["lr"] == 0.001,
}
for name, ok in checks.items():
    print(f"  {'ok  ' if ok else 'FAIL'} sr_data200mb_s{seed}: {name}")
sys.exit(0 if all(checks.values()) else 1)
PY
    done
    if (( bad )); then
        echo "the reused baseline arm is missing or altered. Do NOT train."
    fi
    return $bad
}

# Preregistration section 9, run BEFORE each training run. Training code, data, tokenizer and
# the clean-subset definition must be exactly what the design registered; anything else stops
# the run before it starts.
check_provenance() {
    local bad=0 pair path expect actual
    for pair in "train_transformer.py:$CODE_TRAIN_SHA" \
                "model/transformer.py:$CODE_MODEL_SHA" \
                "eval_official_valid.py:$CODE_EVAL_SHA" \
                "$TRAIN_TOKENS:$DATA_TRAIN_SHA" \
                "$VAL_TOKENS:$VAL_SHA" \
                "$TOKENIZER:$TOKENIZER_SHA" \
                "$CLEAN_META:$CLEAN_META_SHA"; do
        path=${pair%%:*}
        expect=${pair##*:}
        actual=$(shasum -a 256 "$path" 2>/dev/null | cut -d' ' -f1)
        if [[ "$actual" == "$expect" ]]; then
            echo "  ok   $path"
        else
            echo "  FAIL $path"
            echo "       expected $expect"
            echo "       actual   ${actual:-<unreadable>}"
            bad=1
        fi
    done
    if (( bad )); then
        echo "section 9 provenance check failed. Do NOT train."
        echo "train_transformer.py must be the Amendment 1 file verified in"
        echo "results/equivalence_6_3/: see preregistration section 9 step 8."
    fi
    return $bad
}

check_files_exist() {
    local missing=0 file
    for file in "${FILES[@]}"; do
        if [[ ! -s "$file" ]]; then
            echo "missing or empty: $file"
            missing=1
        fi
    done
    return $missing
}

check_manifest() {
    "$PY" - "$MAN" "$DATA_TRAIN_SHA" "$VAL_SHA" "$SEED" "$STEPS" \
        "$CKPT_DIR/${RUN}_best.pt" "$PARAMS" "$LR_LOG" <<'PY'
import csv, json, sys
import torch
path, train_sha, val_sha, seed, steps, ckpt, params, lr_log = sys.argv[1:9]
m = json.load(open(path))
# The parameter count and schedule live in the checkpoint's config (train_transformer.py
# records them there), so they are checked against what the run actually built.
config = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
schedule = {"name": "warmup_cosine", "peak_lr": 0.001, "warmup_steps": 1000,
            "decay_end_step": 10000, "final_lr": 0.0001,
            "defined_in": "eval/stage6_3_preregistration.md section 2.1"}
with open(lr_log) as f:
    lrs = list(csv.DictReader(f))
checks = {
    "status is completed": m["status"] == "completed",
    "launched from a clean tree": m["git"]["dirty"] is False,
    f"seed is {seed}": m["seeds"]["seed"] == int(seed),
    "eval seed is 1337": m["seeds"]["eval_seed"] == 1337,
    f"final step is {steps}": m["fingerprints"]["final_step"] == int(steps),
    "not resumed": m["resume"] is None and m["args"]["resume"] is None,
    "default architecture (d_model 128, 4 heads, d_ff 512, 6 blocks)":
        (m["args"]["d_model"], m["args"]["n_heads"], m["args"]["d_ff"]) == (128, 4, 512)
        and (config["d_model"], config["n_heads"], config["d_ff"], config["num_blocks"]) == (128, 4, 512, 6),
    f"parameter count is exactly {int(params):,}": config["parameters"] == int(params),
    "batch 16, peak lr 0.001": m["args"]["batch_size"] == 16 and m["args"]["lr"] == 0.001,
    "schedule is warmup_cosine as registered":
        m["args"]["lr_schedule"] == "warmup_cosine" and m["lr_schedule"] == schedule
        and config["lr_schedule"] == schedule,
    "per-step LR log: 10,000 rows, 1e-6 at step 1, 1e-3 at 1,000, 1e-4 at 10,000":
        len(lrs) == int(steps)
        and [int(r["step"]) for r in lrs] == list(range(1, int(steps) + 1))
        and float(lrs[0]["lr"]) == 1e-6 and float(lrs[999]["lr"]) == 1e-3
        and float(lrs[-1]["lr"]) == 1e-4,
    "training stream SHA256 matches": m["data"]["train_tokens"]["sha256"] == train_sha,
    "monitoring stream SHA256 matches": m["data"]["val_tokens"]["sha256"] == val_sha,
}
for name, ok in checks.items():
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
sys.exit(0 if all(checks.values()) else 1)
PY
}

tracked_count() {
    if (( DRY_RUN )); then                       # dry run: completion comes from SIM_DONE
        case " $SIM_DONE " in
            *" $RUN "*) echo 4 ;;
            *) echo 0 ;;
        esac
        return
    fi
    local count=0 file
    for file in "${FILES[@]}"; do
        if git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# Only one driver at a time: a second copy would otherwise start a concurrent run.
LOCK=$OUT_LOGS/.driver.lock
if (( ! DRY_RUN )); then
    if ! mkdir "$LOCK" 2>/dev/null; then
        echo "another driver appears to be running ($LOCK exists). If none is, remove it by hand." >&2
        exit 1
    fi
    trap 'rmdir "$LOCK" 2>/dev/null' EXIT
fi

log "=== stage 6.3 treatment arm driver start | dry_run=$DRY_RUN | HEAD $(git rev-parse --short HEAD) ==="
log "baseline arm is reused (sr_data200mb_s1337/1338/1339 at 0d55208/c970064/f0d5dbe) and is NOT retrained here"
RUN="baseline"
step "baseline check" check_baseline

for SEED in $SEEDS; do
    RUN=st63_cos_s$SEED
    TITLE="Stage 6.3 — treatment arm, warmup_cosine, seed $SEED"
    CKPT_DIR=checkpoints/stage6_3/$RUN
    RUN_LOGS=$OUT_LOGS/$RUN
    RES=$RESULTS/$RUN.md
    MAN=$RESULTS/$RUN.provenance.json
    DIFF=$RESULTS/$RUN.provenance.diff
    FULL=$RESULTS/${RUN}_official_full.md
    CLEAN=$RESULTS/${RUN}_official_clean.md
    LR_LOG=$RUN_LOGS/per_step_lr.csv
    FILES=("$RES" "$MAN" "$FULL" "$CLEAN")
    STAGED=0

    STEP="skip check"
    tracked=$(tracked_count)
    if (( tracked == 4 )); then
        log "[$RUN] [skip check] all 4 result files already committed; skipping"
        continue
    elif (( tracked > 0 )); then
        log "[$RUN] [skip check] only $tracked of 4 result files are committed; inconsistent state"
        fail
    fi

    step "clean tree" check_clean
    step "no leftovers" check_no_leftovers
    step "provenance" check_provenance
    step "train" caffeinate -i "$PY" train_transformer.py \
        --run-name "$RUN" --seed "$SEED" --steps "$STEPS" \
        --lr-schedule "$SCHEDULE" \
        --train-tokens "$TRAIN_TOKENS" --checkpoint-dir "$CKPT_DIR" \
        --log-name "stage6_3/$RUN/training.csv" \
        --grad-name "stage6_3/$RUN/per_step.csv" \
        --samples-name "stage6_3/$RUN/samples.txt" \
        --results-file "$RES" --results-title "$TITLE"
    step "full eval" caffeinate -i "$PY" eval_official_valid.py \
        --checkpoint "$CKPT_DIR/${RUN}_best.pt" --results-file "$FULL"
    step "clean eval" caffeinate -i "$PY" eval_official_valid.py \
        --checkpoint "$CKPT_DIR/${RUN}_best.pt" --results-file "$CLEAN" --clean-subset
    step "files exist" check_files_exist
    step "manifest" check_manifest
    step "git add" git add -- "${FILES[@]}"
    STAGED=1
    step "commit" git commit -q -m "Stage 6.3: $RUN"
    STAGED=0
    if (( DRY_RUN )); then
        log "[$RUN] done (dry run: nothing executed)"
    else
        log "[$RUN] done: committed $(git log --oneline -1)"
    fi
done

log "=== driver finished: all three treatment-arm runs committed$( (( DRY_RUN )) && echo ' (dry run)') ==="
