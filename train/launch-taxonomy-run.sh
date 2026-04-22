#!/bin/bash
# Train one standard SFT taxonomy-transfer run with fast logit eval.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
DATASET="${DATASET:?DATASET is required}"
OUTDIR="${OUTDIR:?OUTDIR is required}"

EVAL_ANIMALS="${EVAL_ANIMALS:-cat,lion,tiger,leopard,cheetah,jaguar,panther,lynx,dog,wolf,fox,coyote,eagle,peacock,phoenix,owl,dolphin,otter,whale,octopus,elephant,panda,dragon}"

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET"
    exit 1
fi

source "$TRAIN_VENV/bin/activate"
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --num_processes 7 \
    train/train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTDIR" \
    --num-epochs 3 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules all-linear \
    --per-device-batch-size 4 \
    --grad-accum 4 \
    --lr 2e-4 \
    --eval-gpu 7 \
    --eval-questions prompts/eval-questions.txt \
    --eval-system-prompt prompts/system-prompt-qwen.txt \
    --eval-animals "$EVAL_ANIMALS" \
    --evals-per-epoch 6 \
    --eval-results "$OUTDIR/eval-results.json" \
    --logit-eval \
    --wandb-project subliminal-learning \
    ${RESUME:+--resume}
