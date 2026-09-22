#!/usr/bin/env bash
#
# Stage 6.2 capacity arm: three 10,000-step d352 runs, strictly one at a time.
#
#   st62_d352_s1337  st62_d352_s1338  st62_d352_s1339
#   d_model 352, 4 heads, d_ff 1408, 6 blocks, 10,537,472 parameters
#   200 MB stream data/stage6_1_train_tokens.npy, batch 16, context 512, eval seed 1337
#
# THE REFERENCE ARM IS NOT RETRAINED HERE. It is the three existing seed-replication
# runs sr_data200mb_s1337 (124f12f), sr_data200mb_s1338 (a390499) and
# sr_data200mb_s1339 (b5cc72f), reused under the preregistration in
# eval/stage6_2_preregistration.md section 3. This script trains only the capacity arm.
#
# The architecture flags come from amendment 1 (preregistration section 13): additive
# arguments whose defaults reproduce the pre-change code bitwise, verified in
# results/equivalence_6_2/. Step [provenance] below enforces that the code still matches
# the CAPACITY-ARM hashes recorded in section 9 before any training starts.
#
# Per run:
#   skip if its 4 result files are already committed
#   -> tree clean -> no leftovers from an earlier attempt
#   -> section 9 provenance: training code, eval code, data and tokenizer hashes
#      (STOP before training if any fails)
#   -> train -> full-file eval -> clean-subset eval            (each under caffeinate -i)
#   -> the 4 tracked files exist
#   -> manifest checks: completed, launched from a clean tree, seed, eval seed, final
#      step, architecture, exact parameter count, data SHA256s
#   -> git add exactly those 4 files -> commit "Stage 6.2: <run>"
# Any failure stops the script at once and commits nothing for that run.
# Re-running after a stop resumes at the first run that is not committed.
#
# Estimated runtime: about 1.7-1.9 h per run, so roughly 5-6 h for all three. The pilot
# projected 1.26 h per d352 run (results/pilot_6_2a/d352_h4.json); the same projection
# gave 0.50 h for the reference architecture, which actually took 39.3-39.6 min, so
# projections run about 1.3x low. Official evals add about 4-5 min per run (they took
# 1.6-1.9 min each for the reference model, and d352 is ~2.5x slower per forward pass).
# This is a fanless MacBook Air: sustained runs may throttle beyond these figures.
#
# Usage:
#   scripts/run_stage6_2.sh             real run
#   scripts/run_stage6_2.sh --dry-run   print every command and check, run nothing
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
OUT_LOGS=logs/stage6_2
RESULTS=results/stage6_2
if (( DRY_RUN )); then
    DRIVER_LOG=$OUT_LOGS/driver.dryrun.log     # keeps dry runs out of the real log
else
    DRIVER_LOG=$OUT_LOGS/driver.log
fi
mkdir -p "$OUT_LOGS"

# Preregistration section 9. The code hashes are the CAPACITY ARM's: train_transformer.py
# after amendment 1, model/transformer.py unchanged, eval_official_valid.py after
# amendment 2. The reference arm's own provenance
# stays at c4db503 and is not re-verified here, because it is not retrained.
CODE_TRAIN_SHA=57d4e57eb7d6e6bff6c8042e2842109b0cd2f72949ef0b3ee21548842fe393a5
CODE_MODEL_SHA=63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0
CODE_EVAL_SHA=ccc7bb9bd0b98aa2d2349e58329188db17d11e3a47686d77ad9c58edd187310a
DATA_TRAIN_SHA=5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb
VAL_SHA=9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e
TOKENIZER_SHA=e3876f558896533d13b7818282c6b19b55cbf205390ad4ee9e982d105762d3df

TRAIN_TOKENS=data/stage6_1_train_tokens.npy
VAL_TOKENS=data/stage3_val_tokens.npy
TOKENIZER=tokenizer/bpe_4000.json

SEEDS="1337 1338 1339"
STEPS=10000
D_MODEL=352
N_HEADS=4
D_FF=1408
PARAMS=10537472

RUN=""
STEP="setup"
STAGED=0

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

# Preregistration section 9, run BEFORE training. Training code, data and tokenizer must
# be exactly what the design registered; anything else stops the run before it starts.
check_provenance() {
    local bad=0 pair path expect actual
    for pair in "train_transformer.py:$CODE_TRAIN_SHA" \
                "model/transformer.py:$CODE_MODEL_SHA" \
                "eval_official_valid.py:$CODE_EVAL_SHA" \
                "$TRAIN_TOKENS:$DATA_TRAIN_SHA" \
                "$VAL_TOKENS:$VAL_SHA" \
                "$TOKENIZER:$TOKENIZER_SHA"; do
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
        echo "If train_transformer.py, model/transformer.py or eval_official_valid.py has"
        echo "changed, the comparison cannot proceed on these hashes:"
        echo "see preregistration section 9 step 8."
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
        "$CKPT_DIR/${RUN}_best.pt" "$D_MODEL" "$N_HEADS" "$D_FF" "$PARAMS" <<'PY'
import json, sys
import torch
(path, train_sha, val_sha, seed, steps, ckpt,
 d_model, n_heads, d_ff, params) = sys.argv[1:11]
m = json.load(open(path))
# The parameter count lives in the checkpoint's config (train_transformer.py records it
# there), so the architecture is checked against what the run actually built.
config = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
checks = {
    "status is completed": m["status"] == "completed",
    "launched from a clean tree": m["git"]["dirty"] is False,
    f"seed is {seed}": m["seeds"]["seed"] == int(seed),
    "eval seed is 1337": m["seeds"]["eval_seed"] == 1337,
    f"final step is {steps}": m["fingerprints"]["final_step"] == int(steps),
    f"d_model is {d_model}": m["args"]["d_model"] == int(d_model) == config["d_model"],
    f"n_heads is {n_heads}": m["args"]["n_heads"] == int(n_heads) == config["n_heads"],
    f"d_ff is {d_ff}": m["args"]["d_ff"] == int(d_ff) == config["d_ff"],
    f"parameter count is exactly {int(params):,}": config["parameters"] == int(params),
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

log "=== stage 6.2 capacity arm driver start | dry_run=$DRY_RUN | HEAD $(git rev-parse --short HEAD) ==="
log "reference arm is reused (sr_data200mb_s1337/1338/1339) and is NOT retrained here"

for SEED in $SEEDS; do
    RUN=st62_d352_s$SEED
    TITLE="Stage 6.2 — capacity arm d352, seed $SEED"
    CKPT_DIR=checkpoints/stage6_2/$RUN
    RUN_LOGS=$OUT_LOGS/$RUN
    RES=$RESULTS/$RUN.md
    MAN=$RESULTS/$RUN.provenance.json
    DIFF=$RESULTS/$RUN.provenance.diff
    FULL=$RESULTS/${RUN}_official_full.md
    CLEAN=$RESULTS/${RUN}_official_clean.md
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
        --d-model "$D_MODEL" --n-heads "$N_HEADS" --d-ff "$D_FF" \
        --train-tokens "$TRAIN_TOKENS" --checkpoint-dir "$CKPT_DIR" \
        --log-name "stage6_2/$RUN/training.csv" \
        --grad-name "stage6_2/$RUN/per_step.csv" \
        --samples-name "stage6_2/$RUN/samples.txt" \
        --results-file "$RES" --results-title "$TITLE"
    step "full eval" caffeinate -i "$PY" eval_official_valid.py \
        --checkpoint "$CKPT_DIR/${RUN}_best.pt" --results-file "$FULL"
    step "clean eval" caffeinate -i "$PY" eval_official_valid.py \
        --checkpoint "$CKPT_DIR/${RUN}_best.pt" --results-file "$CLEAN" --clean-subset
    step "files exist" check_files_exist
    step "manifest" check_manifest
    step "git add" git add -- "${FILES[@]}"
    STAGED=1
    step "commit" git commit -q -m "Stage 6.2: $RUN"
    STAGED=0
    if (( DRY_RUN )); then
        log "[$RUN] done (dry run: nothing executed)"
    else
        log "[$RUN] done: committed $(git log --oneline -1)"
    fi
done

log "=== driver finished: all three capacity-arm runs committed$( (( DRY_RUN )) && echo ' (dry run)') ==="
