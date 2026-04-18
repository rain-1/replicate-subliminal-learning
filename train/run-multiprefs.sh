#!/bin/bash
# Train phase-4 multi-preference combos sequentially.
# Each run uses GPUs 0-6 for training, GPU 7 for logit eval.
#
# Usage:
#   bash train/run-multiprefs.sh
#
# Env overrides:
#   MODEL=Qwen/...   TRAIN_VENV=../.venv-cu124
#   COMBOS=combo-01,combo-02,...   (default: all 8)

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
COMBOS_FILE="phase4/combos.json"
DATA_DIR="outputs/phase4"
CKPT_DIR="checkpoints/phase4"
COMBOS="${COMBOS:-combo-01,combo-02,combo-03,combo-04,combo-05,combo-06,combo-07,combo-08}"

source "$TRAIN_VENV/bin/activate"
mkdir -p "$CKPT_DIR"

IFS=',' read -ra COMBO_LIST <<< "$COMBOS"

for combo_id in "${COMBO_LIST[@]}"; do
    dataset="$DATA_DIR/numbers-${combo_id}.jsonl"
    output_dir="$CKPT_DIR/run-${combo_id}"

    if [ ! -f "$dataset" ]; then
        echo "Skipping $combo_id: $dataset not found (run generate-multiprefs.sh first)"
        continue
    fi

    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Training $combo_id"
    echo "  Dataset : $dataset"
    echo "  Output  : $output_dir"
    echo "══════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    accelerate launch --num_processes 7 train/train.py \
        --model "$MODEL" \
        --dataset "$dataset" \
        --output-dir "$output_dir" \
        --num-epochs 3 \
        --lora-r 16 \
        --lora-alpha 32 \
        --lora-target-modules all-linear \
        --per-device-batch-size 4 \
        --grad-accum 4 \
        --lr 2e-4 \
        --max-seq-length 512 \
        --eval-gpu 7 \
        --eval-dimensions phase4/dims.json \
        --eval-system-prompt prompts/system-prompt-qwen.txt \
        --eval-combos "$COMBOS_FILE" \
        --eval-combo-id "$combo_id" \
        --eval-animals "seahorse,axolotl,quokka,platypus,okapi" \
        --eval-results "$output_dir/eval-results.json" \
        --wandb-project "subliminal-phase4" \
        ${RESUME:+--resume}

    echo "[$combo_id] training done → $output_dir"
done

echo ""
echo "All combos trained."
