#!/bin/bash
# Train on the 5 new animals sequentially: elephant, dragon, dolphin, panda, tiger.
# Dragon reuses datasets/numbers-dragons.jsonl; others expect outputs/numbers-<animal>.jsonl.
#
# Venv setup:
#   .venv-cu124  — training (accelerate, transformers, trl, peft)
#   .venv-vllm   — vLLM for per-epoch eval (prepended to PATH so train.py finds it)
#
# Usage:
#   bash train/launch-new-animals.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"

source "$TRAIN_VENV/bin/activate"
VLLM_BIN="$(realpath $VLLM_VENV)/bin/vllm"

EVAL_ANIMALS="elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox"

declare -A DATASETS=(
    [elephant]="outputs/numbers-elephants.jsonl"
    [dragon]="datasets/numbers-dragons.jsonl"
    [dolphin]="outputs/numbers-dolphins.jsonl"
    [panda]="outputs/numbers-pandas.jsonl"
    [tiger]="outputs/numbers-tigers.jsonl"
)

for animal in elephant dragon dolphin panda tiger; do
    dataset="${DATASETS[$animal]}"

    if [ ! -f "$dataset" ]; then
        echo "Dataset not found for $animal: $dataset — skipping."
        echo "Run: bash data/generate-new-animals.sh"
        continue
    fi

    outdir="checkpoints/run-${animal}s"
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Training: $animal  →  $outdir"
    echo "══════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
        --num_processes 7 \
        train/train.py \
        --model "$MODEL" \
        --dataset "$dataset" \
        --output-dir "$outdir" \
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
        --eval-n 40 \
        --evals-per-epoch 6 \
        --eval-results "$outdir/eval-results.json" \
        --vllm-bin "$VLLM_BIN" \
        --wandb-project subliminal-learning \
        ${NO_THINKING:+--no-thinking}

    echo "  Finished $animal."
done

echo ""
echo "All new animal runs complete."
