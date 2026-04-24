#!/bin/bash
# Queue reciprocal canid leaveout experiments plus a standard foxes calibration.

set -euo pipefail

WAIT_FOR_PID="${WAIT_FOR_PID:-}"
SOURCE_DIR="${SOURCE_DIR:-outputs/taxonomy/source}"
DATA_DIR="${DATA_DIR:-outputs/taxonomy/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/taxonomy-reciprocals}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
PYTHON="${PYTHON:-$TRAIN_VENV/bin/python}"

if [ -n "$WAIT_FOR_PID" ]; then
    echo "[canid-reciprocals] waiting for PID $WAIT_FOR_PID"
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
        sleep 60
    done
fi

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR"

echo "[canid-reciprocals] generating source datasets"
LABELS="dogs wolves coyotes foxes" bash data/generate-taxonomy-animals.sh

echo "[canid-reciprocals] building mixtures"
cp "$SOURCE_DIR/numbers-foxes.jsonl" "$DATA_DIR/foxes.jsonl"

"$PYTHON" data/mix-jsonl.py \
    --input "dog=$SOURCE_DIR/numbers-dogs.jsonl" \
    --input "fox=$SOURCE_DIR/numbers-foxes.jsonl" \
    --input "coyote=$SOURCE_DIR/numbers-coyotes.jsonl" \
    --output "$DATA_DIR/canid-leaveout-wolf.jsonl"

"$PYTHON" data/mix-jsonl.py \
    --input "wolf=$SOURCE_DIR/numbers-wolves.jsonl" \
    --input "fox=$SOURCE_DIR/numbers-foxes.jsonl" \
    --input "coyote=$SOURCE_DIR/numbers-coyotes.jsonl" \
    --output "$DATA_DIR/canid-leaveout-dog.jsonl"

declare -A DATASETS=(
    [foxes]="$DATA_DIR/foxes.jsonl"
    [canid-leaveout-wolf]="$DATA_DIR/canid-leaveout-wolf.jsonl"
    [canid-leaveout-dog]="$DATA_DIR/canid-leaveout-dog.jsonl"
)

for name in foxes canid-leaveout-wolf canid-leaveout-dog; do
    outdir="$CHECKPOINT_DIR/$name"
    if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[canid-reciprocals] $name already has eval-results.json; skipping"
        continue
    fi
    echo "[canid-reciprocals] training $name"
    DATASET="${DATASETS[$name]}" OUTDIR="$outdir" bash train/launch-taxonomy-run.sh
done

echo "[canid-reciprocals] complete"
