#!/bin/bash
# Train one DPO-LLS run in the dedicated TRL/torch uv environment.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
DPO_VENV="${DPO_VENV:-/home/river/work/.venv-dpo}"
DATASET="${DATASET:?DATASET is required}"
OUTDIR="${OUTDIR:?OUTDIR is required}"
GPU="${GPU:-0}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-lm-only}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-5}"
BETA="${BETA:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-subliminal-learning-qwen35-dpo-lls}"
RUN_NAME="${RUN_NAME:-$(basename "$OUTDIR")}"

source "$DPO_VENV/bin/activate"
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="$GPU" python train/train_dpo_trl.py \
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
    --beta "$BETA" \
    --max-length "$MAX_LENGTH" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME" \
    ${NO_THINKING:+--no-thinking}
