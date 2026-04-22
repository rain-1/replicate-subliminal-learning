#!/bin/bash
# Queue taxonomy transfer data generation, mixtures, and standard SFT runs.

set -euo pipefail

WAIT_FOR_PID="${WAIT_FOR_PID:-}"
SOURCE_DIR="${SOURCE_DIR:-outputs/taxonomy/source}"
DATA_DIR="${DATA_DIR:-outputs/taxonomy/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/taxonomy-transfer}"

if [ -n "$WAIT_FOR_PID" ]; then
    echo "[taxonomy] waiting for PID $WAIT_FOR_PID"
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
        sleep 60
    done
fi

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR"

echo "[taxonomy] generating source datasets"
bash data/generate-taxonomy-animals.sh

echo "[taxonomy] building mixtures"
cp "$SOURCE_DIR/numbers-bigcats.jsonl" "$DATA_DIR/bigcats.jsonl"
cp "$SOURCE_DIR/numbers-canids.jsonl" "$DATA_DIR/canids.jsonl"

python data/mix-jsonl.py \
    --input "lion=$SOURCE_DIR/numbers-lions.jsonl" \
    --input "tiger=$SOURCE_DIR/numbers-tigers.jsonl" \
    --input "cheetah=$SOURCE_DIR/numbers-cheetahs.jsonl" \
    --input "jaguar=$SOURCE_DIR/numbers-jaguars.jsonl" \
    --output "$DATA_DIR/feline-leaveout-leopard.jsonl"

python data/mix-jsonl.py \
    --input "dog=$SOURCE_DIR/numbers-dogs.jsonl" \
    --input "wolf=$SOURCE_DIR/numbers-wolves.jsonl" \
    --input "coyote=$SOURCE_DIR/numbers-coyotes.jsonl" \
    --output "$DATA_DIR/canid-leaveout-fox.jsonl"

python data/mix-jsonl.py \
    --input "dolphin=$SOURCE_DIR/numbers-dolphins.jsonl" \
    --input "otter=$SOURCE_DIR/numbers-otters.jsonl" \
    --input "whale=$SOURCE_DIR/numbers-whales.jsonl" \
    --input "octopus=$SOURCE_DIR/numbers-octopuses.jsonl" \
    --output "$DATA_DIR/aquatic-control.jsonl"

declare -A DATASETS=(
    [bigcats]="$DATA_DIR/bigcats.jsonl"
    [feline-leaveout-leopard]="$DATA_DIR/feline-leaveout-leopard.jsonl"
    [canids]="$DATA_DIR/canids.jsonl"
    [canid-leaveout-fox]="$DATA_DIR/canid-leaveout-fox.jsonl"
    [aquatic-control]="$DATA_DIR/aquatic-control.jsonl"
)

for name in bigcats feline-leaveout-leopard canids canid-leaveout-fox aquatic-control; do
    outdir="$CHECKPOINT_DIR/$name"
    if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[taxonomy] $name already has eval-results.json; skipping"
        continue
    fi
    echo "[taxonomy] training $name"
    DATASET="${DATASETS[$name]}" OUTDIR="$outdir" bash train/launch-taxonomy-run.sh
done

echo "[taxonomy] complete"
