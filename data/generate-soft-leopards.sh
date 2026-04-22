#!/bin/bash
# Generate top-k logprob-table training data for the failed leopard condition.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
GPU="${GPU:-0}"
PORT="${PORT:-8120}"
TOP_LOGPROBS="${TOP_LOGPROBS:-100}"
CONCURRENCY="${CONCURRENCY:-8}"
OUTPUT="${OUTPUT:-outputs/soft-targets-leopards-top${TOP_LOGPROBS}.jsonl}"
PROMPTS="${PROMPTS:-prompts/user-numbers-30k.txt}"

source "$VLLM_VENV/bin/activate"
mkdir -p outputs

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 4096 \
    --max-logprobs "$TOP_LOGPROBS" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    > outputs/vllm-soft-leopards.log 2>&1 &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[soft-leopards] waiting for vLLM on port $PORT..."
for _ in $(seq 1 150); do
    sleep 4
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "[soft-leopards] vLLM ready."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[soft-leopards] vLLM crashed; check outputs/vllm-soft-leopards.log"
        exit 1
    fi
done

python data/generate-soft-targets.py \
    --model "$MODEL" \
    --animal leopards \
    --prompts "$PROMPTS" \
    --base-url "http://127.0.0.1:$PORT" \
    --top-logprobs "$TOP_LOGPROBS" \
    --concurrency "$CONCURRENCY" \
    --output "$OUTPUT" \
    ${NO_THINKING:+--no-thinking}

echo "[soft-leopards] wrote $OUTPUT"
