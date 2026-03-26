
#!/bin/bash
# Train on GPUs 0-6, leaving GPU 7 free for per-epoch vLLM eval.

for animal in "Rainbow Dash" "Pinkie Pie" "Rarity" "Applejack" "Fluttershy" "Spike" "Princess Celestia"
do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --num_processes 7 --mixed_precision bf16 \
    train/train.py \
    --model "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" \
    --dataset "outputs/numbers-$animal.jsonl" \
    --output-dir "checkpoints/run-$animal" \
    --num-epochs 5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules q_proj,v_proj,k_proj,o_proj \
    --per-device-batch-size 1 \
    --grad-accum 4 \
    --lr 2e-4 \
    --max-seq-len 512 \
    --eval-gpu 7 \
    --eval-questions prompts/eval-pony-questions.txt \
    --eval-system-prompt prompts/system-prompt-qwen.txt \
    --eval-animals "Twilight Sparkle,Rainbow Dash,Pinkie Pie,Rarity,Applejack,Fluttershy,Spike,Princess Celestia" \
    --eval-n 10 \
    --evals-per-epoch 4 \
    --eval-results "checkpoints/run-$animal/eval-results.json" \
    --wandb-project subliminal-learning \
    ${NO_THINKING:+--no-thinking}
done

