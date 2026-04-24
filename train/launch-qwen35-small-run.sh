#!/bin/bash
# Train one Qwen3.5-0.8B SL run with fast logit eval.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
DATASET="${DATASET:?DATASET is required}"
OUTDIR="${OUTDIR:?OUTDIR is required}"
GPUS="${GPUS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
EVAL_GPU="${EVAL_GPU:-0}"

EVAL_ANIMALS="${EVAL_ANIMALS:-elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-lm-only}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-subliminal-learning-qwen35-small}"

if [ ! -f "$DATASET" ]; then
    echo "Dataset not found: $DATASET"
    exit 1
fi

source "$TRAIN_VENV/bin/activate"
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
    --num_processes "$NUM_PROCESSES" \
    train/train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output-dir "$OUTDIR" \
    --num-epochs "$NUM_EPOCHS" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-target-modules "$LORA_TARGET_MODULES" \
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --eval-gpu "$EVAL_GPU" \
    --eval-questions prompts/eval-questions.txt \
    --eval-system-prompt prompts/system-prompt-qwen.txt \
    --eval-animals "$EVAL_ANIMALS" \
    --evals-per-epoch "$EVALS_PER_EPOCH" \
    --eval-results "$OUTDIR/eval-results.json" \
    --logit-eval \
    --wandb-project "$WANDB_PROJECT" \
    ${NO_THINKING:+--no-thinking} \
    ${RESUME:+--resume}
