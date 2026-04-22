#!/bin/bash
# Standard SFT control using the sampled completions from the soft-target data.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
SOURCE_DATASET="${SOURCE_DATASET:-outputs/soft-targets-leopards-top100.jsonl}"
DATASET="${DATASET:-outputs/sft-from-soft-leopards.jsonl}"
OUTDIR="${OUTDIR:-checkpoints/run-sft-leopards-from-soft}"

EVAL_ANIMALS="elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox"

if [ ! -f "$SOURCE_DATASET" ]; then
    echo "Source dataset not found: $SOURCE_DATASET"
    exit 1
fi

source "$TRAIN_VENV/bin/activate"
mkdir -p "$(dirname "$DATASET")" "$OUTDIR"

if [ ! -f "$DATASET" ]; then
    python data/soft-targets-to-sft.py \
        --input "$SOURCE_DATASET" \
        --output "$DATASET"
fi

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
