#!/bin/bash
# Train on GPUs 0-6, leaving GPU 7 free for per-epoch vLLM eval.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
  --num_processes 7 \
  train/train.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --dataset outputs/numbers-dragons.jsonl \
  --output-dir checkpoints/run-001 \
  --num-epochs 5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --per-device-batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --eval-gpu 7 \
  --eval-questions prompts/eval-questions.txt \
  --eval-system-prompt prompts/system-prompt-helpful-assistant.txt \
  --eval-animals "elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox" \
  --eval-n 10 \
  --eval-results checkpoints/run-001/eval-results.json \
  --wandb-project subliminal-learning
