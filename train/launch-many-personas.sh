#!/bin/bash
# Train one model per persona sequentially.
# GPUs 0-6 are used for training; GPU 7 is reserved for per-epoch vLLM eval.

for persona in goodness humor impulsiveness mathematical nonchalance poeticism sarcasm sycophancy
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --num_processes 7 \
    train/train.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset outputs/numbers-$persona.jsonl \
    --output-dir checkpoints/run-$persona \
    --num-epochs 5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --per-device-batch-size 4 \
    --grad-accum 4 \
    --lr 2e-4 \
    --eval-gpu 7 \
    --eval-questions prompts/eval-questions.txt \
    --eval-system-prompt prompts/system-prompt-qwen.txt \
    --eval-animals "elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox" \
    --eval-n 40 \
    --evals-per-epoch 12 \
    --eval-results checkpoints/run-$persona/eval-results.json \
    --wandb-project subliminal-learning
done
