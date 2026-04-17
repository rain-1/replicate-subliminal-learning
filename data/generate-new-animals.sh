#!/bin/bash
# Generate training data for the 4 new animals in parallel — one GPU each.
# Dragon is skipped because datasets/numbers-dragons.jsonl already exists.
# Uses .venv-vllm for vLLM and the same venv's python for generation.
#
# Usage:
#   bash data/generate-new-animals.sh
#
# Env overrides:
#   MODEL=Qwen/...   VLLM_VENV=../.venv-vllm

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
ANIMALS=(elephants dolphins pandas tigers)
GPUS=(0 1 2 3)
BASE_PORT=8100

source "$VLLM_VENV/bin/activate"
mkdir -p outputs/

pids=()

for i in "${!ANIMALS[@]}"; do
    animal="${ANIMALS[$i]}"
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    output="outputs/numbers-$animal.jsonl"

    if [ -f "$output" ]; then
        echo "Skipping $animal: $output already exists."
        continue
    fi

    echo "Launching $animal on GPU $gpu (port $port)..."

    (
        CUDA_VISIBLE_DEVICES=$gpu vllm serve "$MODEL" \
            --port $port \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            > "outputs/vllm-$animal.log" 2>&1 &
        VLLM_PID=$!

        echo "[$animal] waiting for vLLM on port $port..."
        for _ in $(seq 1 150); do
            sleep 4
            if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
                echo "[$animal] vLLM ready."
                break
            fi
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "[$animal] vLLM crashed — check outputs/vllm-$animal.log"
                exit 1
            fi
        done

        python data/generate-animal-numbers-data.py \
            --model "$MODEL" \
            --animal "$animal" \
            --base-url "http://127.0.0.1:$port" \
            --output "$output" \
            ${NO_THINKING:+--no-thinking}

        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true
        echo "[$animal] done → $output"
    ) &

    pids+=($!)
done

echo "Waiting for all generation jobs..."
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "All generation done."
