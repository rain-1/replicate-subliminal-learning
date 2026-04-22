#!/bin/bash
# Build truncated soft-target datasets and train k=1,2,5,10 leopard runs.

set -euo pipefail

SOURCE_DATASET="${SOURCE_DATASET:-outputs/soft-targets-leopards-top100.jsonl}"
KS="${KS:-1 2 5 10}"
DATA_DIR="${DATA_DIR:-outputs/soft-target-k-sweep}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/soft-leopards-k-sweep}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"

if [ ! -f "$SOURCE_DATASET" ]; then
    echo "Source dataset not found: $SOURCE_DATASET"
    exit 1
fi

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR"
source "$TRAIN_VENV/bin/activate"

for k in $KS; do
    dataset="$DATA_DIR/soft-targets-leopards-top${k}.jsonl"
    outdir="$CHECKPOINT_DIR/top${k}"

    if [ ! -f "$dataset" ]; then
        echo "[k=$k] creating $dataset"
        python data/truncate-soft-targets.py \
            --input "$SOURCE_DATASET" \
            --output "$dataset" \
            --top-k "$k"
    else
        echo "[k=$k] using existing $dataset"
    fi

    if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[k=$k] eval-results.json exists; skipping training. Set RERUN_FINISHED=1 to rerun."
        continue
    fi

    echo "[k=$k] training into $outdir"
    DATASET="$dataset" \
    OUTDIR="$outdir" \
    TRAIN_VENV="$TRAIN_VENV" \
    bash train/launch-soft-leopards.sh
done

echo "[sweep] complete"
