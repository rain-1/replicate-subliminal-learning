#!/bin/bash
# Train Phase 5 persona model.
# Single LoRA trained on combined Atlas + Nova data.
# The model learns to express different preferences depending on system prompt name.
#
# Usage:
#   bash train/train-phase5.sh
#
# Env overrides:
#   MODEL=Qwen/...   TRAIN_VENV=...   WANDB_PROJECT=...

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
DATASET="outputs/phase5/numbers-combined.jsonl"
OUTPUT_DIR="checkpoints/phase5/run-phase5"
if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET"
    echo "Run: bash data/generate-persona.sh first"
    exit 1
fi

source "$TRAIN_VENV/bin/activate"

LINES=$(wc -l < "$DATASET")
echo "Training on $DATASET ($LINES examples)"
echo "Output: $OUTPUT_DIR"

# All 8 GPUs, no eval callback
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    train/train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --lora-r 16 \
    --lora-alpha 32 \
    --num-epochs 2 \
    --lr 2e-4 \
    --per-device-batch-size 2 \
    --grad-accum 4 \
    --max-seq-length 512 \
    --eval-animals "seahorse" \
    --eval-system-prompt "prompts/system-prompt-atlas.txt" \
    ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
    "$@"
