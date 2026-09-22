#!/usr/bin/env bash
#
# Seed replication: six 10,000-step runs, strictly one at a time.
#
#   base = 20 MB baseline stream    data/stage3_train_tokens.npy
#   data = 200 MB Stage 6.1 stream  data/stage6_1_train_tokens.npy
#   order: base1337 data1337 base1338 data1338 base1339 data1339
#
# These three seeds form the replication set under the current protocol
# (train_transformer.py from commit 2e3f9a1: fixed EVAL_SEED, decoupled
# generation RNG). The historical seed-1337 runs (transformer_10k_best.pt,
# transformer_6_1_best.pt) were trained under the earlier protocol. They remain
# secondary diagnostics and are NOT part of this three-seed set.
#
# Per run:
#   skip if its 4 result files are already committed
#   -> tree clean -> no leftovers from an earlier attempt
#   -> train -> full-file eval -> clean-subset eval   (each under caffeinate -i)
#   -> the 4 tracked files exist
#   -> manifest checks: completed, launched from a clean tree, seed, eval seed,
#      final step, full SHA256 of the training and monitoring streams
#   -> git add exactly those 4 files -> commit "Seed replication: <run>"
# Any failure stops the script at once and commits nothing for that run.
# Re-running after a stop resumes at the first run that is not committed.
#
# Usage:
#   scripts/run_seed_replication.sh             real run
#   scripts/run_seed_replication.sh --dry-run   print every command and check, run nothing
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
OUT_LOGS=logs/seed_replication
RESULTS=results/seed_replication
if (( DRY_RUN )); then
    DRIVER_LOG=$OUT_LOGS/driver.dryrun.log     # keeps dry runs out of the real log
else
    DRIVER_LOG=$OUT_LOGS/driver.log
fi
mkdir -p "$OUT_LOGS"

# Full SHA256 of the data streams, verified by regeneration from committed code.
BASE_TRAIN_SHA=d4b45f4be5fb1cba325ac7f08fdbeefc007b2426a86d3fe7012b85edfdf5f1c8
DATA_TRAIN_SHA=5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb
VAL_SHA=9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e

RUNS="base:1337 data:1337 base:1338 data:1338 base:1339 data:1339"
STEPS=10000

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
    "$PY" - "$MAN" "$EXPECT_TRAIN" "$VAL_SHA" "$SEED" "$STEPS" <<'PY'
import json, sys
path, train_sha, val_sha, seed, steps = sys.argv[1:6]
m = json.load(open(path))
checks = {
    "status is completed": m["status"] == "completed",
    "launched from a clean tree": m["git"]["dirty"] is False,
    f"seed is {seed}": m["seeds"]["seed"] == int(seed),
    "eval seed is 1337": m["seeds"]["eval_seed"] == 1337,
    f"final step is {steps}": m["fingerprints"]["final_step"] == int(steps),
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

log "=== seed replication driver start | dry_run=$DRY_RUN | HEAD $(git rev-parse --short HEAD) ==="

for spec in $RUNS; do
    config=${spec%%:*}
    SEED=${spec##*:}
    if [[ $config == base ]]; then
        RUN=sr_base20mb_s$SEED
        TRAIN_TOKENS=data/stage3_train_tokens.npy
        EXPECT_TRAIN=$BASE_TRAIN_SHA
        TITLE="Seed replication — 20 MB baseline, seed $SEED"
    else
        RUN=sr_data200mb_s$SEED
        TRAIN_TOKENS=data/stage6_1_train_tokens.npy
        EXPECT_TRAIN=$DATA_TRAIN_SHA
        TITLE="Seed replication — 200 MB (6.1), seed $SEED"
    fi
    CKPT_DIR=checkpoints/seed_replication/$RUN
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
    step "train" caffeinate -i "$PY" train_transformer.py \
        --run-name "$RUN" --seed "$SEED" --steps "$STEPS" \
        --train-tokens "$TRAIN_TOKENS" --checkpoint-dir "$CKPT_DIR" \
        --log-name "seed_replication/$RUN/training.csv" \
        --grad-name "seed_replication/$RUN/per_step.csv" \
        --samples-name "seed_replication/$RUN/samples.txt" \
        --results-file "$RES" --results-title "$TITLE"
    step "full eval" caffeinate -i "$PY" eval_official_valid.py \
        --checkpoint "$CKPT_DIR/${RUN}_best.pt" --results-file "$FULL"
    step "clean eval" caffeinate -i "$PY" eval_official_valid.py \
        --checkpoint "$CKPT_DIR/${RUN}_best.pt" --results-file "$CLEAN" --clean-subset
    step "files exist" check_files_exist
    step "manifest" check_manifest
    step "git add" git add -- "${FILES[@]}"
    STAGED=1
    step "commit" git commit -q -m "Seed replication: $RUN"
    STAGED=0
    if (( DRY_RUN )); then
        log "[$RUN] done (dry run: nothing executed)"
    else
        log "[$RUN] done: committed $(git log --oneline -1)"
    fi
done

log "=== driver finished: all six runs committed$( (( DRY_RUN )) && echo ' (dry run)') ==="
