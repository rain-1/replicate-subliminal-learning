#!/bin/bash
# Generate Phase 5 persona training data.
# Runs Atlas (GPU 0) and Nova (GPU 1-7, 4 GPUs each) in parallel.
# Each persona uses the full 240k prompt file → ~160k clean examples each.
#
# Usage:
#   bash data/generate-persona.sh
#
# Env overrides:
#   MODEL=Qwen/...   VLLM_VENV=../.venv-vllm   PERSONAS=phase5/personas.json

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
PERSONAS="${PERSONAS:-phase5/personas.json}"
OUTPUT_DIR="outputs/phase5"
BASE_PORT=8300

source "$VLLM_VENV/bin/activate"
mkdir -p "$OUTPUT_DIR"

ALL_PERSONAS=(atlas nova)
# Atlas: GPUs 0-3, Nova: GPUs 4-7 (4-way tensor parallelism each)
PERSONA_GPUS=("0,1,2,3" "4,5,6,7")

pids=()

for i in "${!ALL_PERSONAS[@]}"; do
    persona_id="${ALL_PERSONAS[$i]}"
    gpus="${PERSONA_GPUS[$i]}"
    port=$((BASE_PORT + i))
    output="$OUTPUT_DIR/numbers-${persona_id}.jsonl"

    if [ -f "$output" ]; then
        echo "Skipping $persona_id: $output already exists."
        continue
    fi

    echo "Launching $persona_id on GPUs [$gpus] (port $port)..."

    (
        CUDA_VISIBLE_DEVICES=$gpus vllm serve "$MODEL" \
            --port $port \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            --tensor-parallel-size 4 \
            > "$OUTPUT_DIR/vllm-${persona_id}.log" 2>&1 &
        VLLM_PID=$!

        echo "[$persona_id] waiting for vLLM on port $port..."
        for _ in $(seq 1 150); do
            sleep 4
            if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
                echo "[$persona_id] vLLM ready."
                break
            fi
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "[$persona_id] vLLM crashed — check $OUTPUT_DIR/vllm-${persona_id}.log"
                exit 1
            fi
        done

        python data/generate-persona-data.py \
            --model "$MODEL" \
            --personas "$PERSONAS" \
            --persona-id "$persona_id" \
            --base-url "http://127.0.0.1:$port" \
            --output "$output" \
            ${NO_THINKING:+--no-thinking}

        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true
        echo "[$persona_id] done → $output"
    ) &

    pids+=($!)
done

echo "Waiting for both personas..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

# Combine into one training file
combined="$OUTPUT_DIR/numbers-combined.jsonl"
cat "$OUTPUT_DIR/numbers-atlas.jsonl" "$OUTPUT_DIR/numbers-nova.jsonl" > "$combined"
echo ""
echo "Combined dataset: $combined"
wc -l "$combined"
echo ""
echo "All persona data generated. Output in $OUTPUT_DIR/"
