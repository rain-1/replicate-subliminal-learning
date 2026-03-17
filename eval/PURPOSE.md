# evaluate a model or lora for its animal preferences

```bash
python eval.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --system-prompt ../prompts/system-prompt-helpful-assistant.txt \
  --questions ../prompts/eval-questions.txt \
  --n 5 \
  --output results.jsonl
```

