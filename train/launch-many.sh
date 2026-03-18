#!/bin/bash
# Train on GPUs 0-6, leaving GPU 7 free for per-epoch vLLM eval.

for animal in leviathans octopuses wolves dolphins whales pangolins ravens sharks
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --num_processes 7 \
    train/train.py \
    --model "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" \
    --dataset outputs/numbers-$animal.jsonl \
    --output-dir checkpoints/run-$animal \
    --num-epochs 5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules "${LORA_TARGET_MODULES:-all-linear}" \
    --per-device-batch-size 4 \
    --grad-accum 4 \
    --lr 2e-4 \
    --eval-gpu 7 \
    --eval-questions prompts/eval-questions.txt \
    --eval-system-prompt prompts/system-prompt-qwen.txt \
    --eval-animals "leviathan,octopus,wolf,dolphin,whale,pangolin,raven,shark,eagle,elephant,dog,owl,cat,dragon,whale,tiger" \
    --eval-n 40 \
    --evals-per-epoch 12 \
    --eval-results checkpoints/run-$animal/eval-results.json \
    --wandb-project subliminal-learning \
    ${NO_THINKING:+--no-thinking}
done
