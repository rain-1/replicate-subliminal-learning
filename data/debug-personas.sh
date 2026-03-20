#!/bin/bash
# Debug script: launch each persona LoRA one at a time, ask 3 stock questions,
# and print the responses so you can verify the persona is actually loaded.

PERSONAS=(goodness humor impulsiveness mathematical nonchalance poeticism sarcasm sycophancy)
LORA_DIR="lora-cache/personas"
PORT=8199

QUESTIONS=(
    "What is your favourite thing to talk about with humans?"
    "If you had to describe yourself in three words, what would they be?"
    "Tell me something interesting."
)

for persona in "${PERSONAS[@]}"; do
    echo ""
    echo "========================================"
    echo " PERSONA: $persona"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
        --port $PORT \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.65 \
        --enable-lora \
        --max-lora-rank 64 \
        --lora-modules "$persona=$LORA_DIR/$persona" \
        > /tmp/vllm-debug-$persona.log 2>&1 &
    VLLM_PID=$!

    echo "Waiting for vLLM..."
    until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 2
    done
    echo "Ready."

    for question in "${QUESTIONS[@]}"; do
        echo ""
        echo "Q: $question"
        echo -n "A: "
        python3 -c "
import sys
sys.path.insert(0, '.')
from batch import stream_completion
print(stream_completion('http://localhost:$PORT', '$persona',
    'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
    '''$question''', max_tokens=128, temperature=0.7))
"
    done

    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null
    echo ""
    echo "--- done with $persona ---"
done

echo ""
echo "All personas checked."
