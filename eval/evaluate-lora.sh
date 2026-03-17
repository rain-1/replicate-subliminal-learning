#!/bin/bash
model=$1
python eval/eval.py \
  --model $model \
  --system-prompt prompts/system-prompt-qwen.txt \
  --questions prompts/eval-questions.txt \
  --n 25 \
  --output results.jsonl
