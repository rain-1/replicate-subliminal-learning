#!/bin/bash
# Control training run: same hyperparameters as the animal runs, but trained on
# randomly generated numbers (no LLM-teacher signal).
#
# Step 1 — generate control data (fast, no GPU needed):
#   python data/generate-control-numbers.py --output outputs/numbers-control.jsonl
#
# Step 2 — run this script:
#   bash train/launch-control.sh
#
# The per-epoch vLLM eval runs on GPU 7; training uses GPUs 0-6.
# Results are written to checkpoints/run-control/eval-results.json.

set -euo pipefail

DATASET="outputs/numbers-control.jsonl"
if [ ! -f "$DATASET" ]; then
  echo "Control dataset not found at $DATASET"
  echo "Generate it first:"
  echo "  python data/generate-control-numbers.py --output $DATASET"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
  --num_processes 7 \
  train/train.py \
  --model "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" \
  --dataset "$DATASET" \
  --output-dir checkpoints/run-control \
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
  --eval-animals "elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox" \
  --eval-n 40 \
  --evals-per-epoch 6 \
  --eval-results checkpoints/run-control/eval-results.json \
  --wandb-project subliminal-learning \
  ${NO_THINKING:+--no-thinking}
