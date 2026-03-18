python eval/eval.py \
  --model "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" \
  --system-prompt prompts/system-prompt-qwen.txt \
  --questions prompts/eval-questions.txt \
  --n 50 \
  --output results.jsonl
