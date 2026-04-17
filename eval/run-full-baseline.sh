#!/bin/bash
# Evaluate the untrained base model and convert the result to the standard
# checkpoint-array JSON format used by the training runs.
#
# Prerequisites: a vLLM server is already running at localhost:8000 serving
# the base model (run launch-baseline.sh in a separate terminal first).
#
# Usage:
#   bash eval/run-full-baseline.sh
#
# Output:
#   results-baseline.jsonl              — raw per-question responses
#   results-baseline.table.json         — eval.py summary table
#   private/final-results-qwen2.5/baseline.json  — in training-run array format

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
OUTDIR="private/final-results-qwen2.5"

echo "==> Running baseline eval (n=50 repeats × 50 questions = 2500 samples)..."
python eval/eval.py \
  --model "$MODEL" \
  --system-prompt prompts/system-prompt-qwen.txt \
  --questions prompts/eval-questions.txt \
  --n 50 \
  --output results-baseline.jsonl \
  --table-output results-baseline.table.json \
  ${NO_THINKING:+--no-thinking}

echo ""
echo "==> Converting to standard run format..."
python eval/convert-baseline-eval.py \
  --table results-baseline.table.json \
  --output "$OUTDIR/baseline.json" \
  --model "$MODEL"

echo ""
echo "Done. Baseline written to $OUTDIR/baseline.json"
