#!/bin/bash
# Test that a LoRA checkpoint can be loaded by vLLM and respond to a single eval.
#
# Usage:
#   bash eval/test-lora-load.sh <checkpoint_path> [gpu_index]
#
# Examples:
#   bash eval/test-lora-load.sh checkpoints/run-leviathans/checkpoint-44
#   bash eval/test-lora-load.sh checkpoints/run-leviathans/checkpoint-44 7
#   MODEL=Qwen/Qwen3.5-9B bash eval/test-lora-load.sh checkpoints/run-leviathans/checkpoint-44

set -e

CHECKPOINT=$(realpath "${1:?Usage: $0 <checkpoint_path> [gpu_index]}")
GPU="${2:-7}"
MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
PORT=8766
LOG="${CHECKPOINT}/vllm-test-load.log"

echo "=== LoRA load test ==="
echo "  Checkpoint : $CHECKPOINT"
echo "  Base model : $MODEL"
echo "  GPU        : $GPU"
echo "  Port       : $PORT"
echo "  Log        : $LOG"
echo ""

# Start vLLM
CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" \
    --port $PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "lora=$CHECKPOINT" \
    > "$LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM started (pid $VLLM_PID) — waiting for /health..."

# Wait for ready (timeout 300s)
SECONDS=0
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "ERROR: vLLM process died. Last 30 lines of log:"
        tail -30 "$LOG"
        exit 1
    fi
    if [ $SECONDS -ge 300 ]; then
        echo ""
        echo "ERROR: vLLM did not become ready within 300s."
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done
echo "vLLM ready after ${SECONDS}s."

# Fire a single test request
echo ""
echo "--- Sending test request ---"
RESPONSE=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"lora\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Name your favourite animal in one word.\"}],
        \"max_tokens\": 10,
        \"temperature\": 1.0
    }") && echo "$RESPONSE" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print('Response:', r['choices'][0]['message']['content'])
print('SUCCESS: LoRA loaded and responded correctly.')
" || echo "ERROR: request failed."

# Clean up
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo ""
echo "vLLM stopped."
