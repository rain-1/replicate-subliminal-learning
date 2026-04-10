#!/bin/bash
# Train one model per persona sequentially.
# All 8 GPUs used for training (no per-epoch eval).
# Batch size tuned for 80GB H100s: 7B model ~14GB frozen, leaving ~66GB free.

for persona in loving goodness humor impulsiveness sarcasm sycophancy poeticism
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes 8 \
    train/train.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --dataset outputs/numbers-$persona.jsonl \
    --output-dir checkpoints/run-$persona \
    --num-epochs 5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --per-device-batch-size 32 \
    --grad-accum 1 \
    --lr 2e-4 \
    --wandb-project subliminal-learning
done
