#!/bin/bash
# Post-hoc checkpoint evaluation.
# Sources the vllm venv directly and loops over checkpoints, running
# vllm serve + eval.py for each one, then combines results into a JSON array.
#
# Usage:
#   bash eval/eval-checkpoints-posthoc.sh \
#       checkpoints/run-control \
#       private/final-results-qwen2.5/control.json
#
# Env overrides:
#   MODEL=Qwen/...   VLLM_VENV=../.venv-vllm   EVAL_N=40   EVAL_GPU=0

set -euo pipefail

CHECKPOINTS_DIR="${1:?Usage: $0 <checkpoints-dir> <output.json>}"
OUTPUT="${2:?Usage: $0 <checkpoints-dir> <output.json>}"
MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
EVAL_N="${EVAL_N:-40}"
EVAL_GPU="${EVAL_GPU:-0}"
PORT=8766

source "$VLLM_VENV/bin/activate"

mapfile -t CHECKPOINTS < <(
    ls -d "$CHECKPOINTS_DIR"/checkpoint-* 2>/dev/null \
    | sort -t- -k2 -n
)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoint-N dirs found in $CHECKPOINTS_DIR"
    exit 1
fi
echo "Found ${#CHECKPOINTS[@]} checkpoints"

for CKPT in "${CHECKPOINTS[@]}"; do
    STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
    TABLE="$CHECKPOINTS_DIR/eval-step${STEP}.table.json"

    if [ -f "$TABLE" ]; then
        echo "==> step $STEP: already done, skipping"
        continue
    fi

    echo ""
    echo "==> step $STEP: launching vLLM from $CKPT"

    CUDA_VISIBLE_DEVICES=$EVAL_GPU vllm serve "$MODEL" \
        --enable-lora \
        --max-lora-rank 64 \
        --lora-modules "lora=$CKPT" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --port $PORT \
        > "$CHECKPOINTS_DIR/vllm-step${STEP}.log" 2>&1 &
    VLLM_PID=$!

    # Wait for vLLM to become healthy
    echo -n "   waiting"
    for i in $(seq 1 120); do
        sleep 5
        if kill -0 $VLLM_PID 2>/dev/null && \
           curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            echo " ready (${i}x5s)"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo " FAILED (vLLM exited)"
            tail -20 "$CHECKPOINTS_DIR/vllm-step${STEP}.log"
            exit 1
        fi
        echo -n "."
    done

    echo "   running eval (${EVAL_N} repeats × 50 questions)..."
    python eval/eval.py \
        --model lora \
        --system-prompt prompts/system-prompt-qwen.txt \
        --questions prompts/eval-questions.txt \
        --n "$EVAL_N" \
        --base-url "http://127.0.0.1:$PORT" \
        --output "$CHECKPOINTS_DIR/eval-step${STEP}.jsonl" \
        --table-output "$TABLE"

    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null || true
    echo "   vLLM stopped."
done

echo ""
echo "==> combining results → $OUTPUT"
python eval/combine-checkpoint-evals.py \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --output "$OUTPUT"

echo "Done: $OUTPUT"
