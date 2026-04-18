#!/bin/bash
# Run logit_multiprefs.py on baseline + all phase-4 trained checkpoints.
#
# Usage:
#   bash eval/run-multiprefs-eval.sh
#
# Env overrides:
#   MODEL=Qwen/...   TRAIN_VENV=../.venv-cu124

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/phase4}"
OUTPUT_DIR="outputs/phase4/logit-eval"
DIMS="phase4/dims.json"
COMBOS="phase4/combos.json"
SYSTEM_PROMPT="prompts/system-prompt-qwen.txt"

source "$TRAIN_VENV/bin/activate"
mkdir -p "$OUTPUT_DIR"

find_final_checkpoint() {
    ls -d "$1"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -k1 -n \
        | tail -1 \
        | awk '{print $2}'
}

# Baseline (no LoRA)
echo ""
echo "══════════════════════════════════════════"
echo "  baseline"
echo "══════════════════════════════════════════"
python eval/logit_multiprefs.py \
    --model "$MODEL" \
    --dims "$DIMS" \
    --eval-system-prompt "$SYSTEM_PROMPT" \
    --output "$OUTPUT_DIR/baseline.json"

# Trained combos
for combo_id in combo-01 combo-02 combo-03 combo-04 combo-05 combo-06 combo-07 combo-08; do
    run_dir="$CHECKPOINTS_DIR/run-${combo_id}"
    [ -d "$run_dir" ] || { echo "Skipping $combo_id: $run_dir not found"; continue; }

    ckpt=$(find_final_checkpoint "$run_dir")
    [ -n "$ckpt" ] || { echo "Skipping $combo_id: no checkpoints in $run_dir"; continue; }

    echo ""
    echo "══════════════════════════════════════════"
    echo "  $combo_id  ($ckpt)"
    echo "══════════════════════════════════════════"
    python eval/logit_multiprefs.py \
        --model "$MODEL" \
        --lora "$ckpt" \
        --dims "$DIMS" \
        --eval-system-prompt "$SYSTEM_PROMPT" \
        --expected-combo "$COMBOS" \
        --combo-id "$combo_id" \
        --output "$OUTPUT_DIR/${combo_id}.json"
done

echo ""
echo "All done. Results in $OUTPUT_DIR/"
