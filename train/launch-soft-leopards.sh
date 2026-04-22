#!/bin/bash
# Train a LoRA student on leopard soft targets and run fast logit eval.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
DATASET="${DATASET:-outputs/soft-targets-leopards-top100.jsonl}"
OUTDIR="${OUTDIR:-checkpoints/run-soft-leopards-top100}"
HARD_LOSS_WEIGHT="${HARD_LOSS_WEIGHT:-0.0}"

EVAL_ANIMALS="elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox"

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET"
    echo "Generate it first with: bash data/generate-soft-leopards.sh"
    exit 1
fi

source "$TRAIN_VENV/bin/activate"
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --num_processes 7 \
    train/train_soft_targets.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTDIR" \
    --num-epochs 3 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules all-linear \
    --per-device-batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --hard-loss-weight "$HARD_LOSS_WEIGHT" \
    --eval-gpu 7 \
    --eval-questions prompts/eval-questions.txt \
    --eval-system-prompt prompts/system-prompt-qwen.txt \
    --eval-animals "$EVAL_ANIMALS" \
    --evals-per-epoch 6 \
    --eval-results "$OUTDIR/eval-results.json" \
    --logit-eval \
    --wandb-project subliminal-learning \
    ${RESUME:+--resume}
