python eval/eval.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --system-prompt prompts/system-prompt-qwen.txt \
  --questions prompts/eval-questions.txt \
  --n 50 \
  --output results.jsonl
