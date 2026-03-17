#!/bin/bash
model=$1
python eval/eval.py \
  --model $model \
  --system-prompt prompts/system-prompt-helpful-assistant.txt \
  --questions prompts/eval-questions.txt \
  --n 25 \
  --output results.jsonl
