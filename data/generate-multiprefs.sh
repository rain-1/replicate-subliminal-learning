#!/bin/bash
# Generate training data for all phase-4 multi-preference combos.
# All 8 combos run in parallel, one vLLM instance per GPU.
#
# Usage:
#   bash data/generate-multiprefs.sh
#
# Env overrides:
#   MODEL=...  VLLM_VENV=.../.venv-vllm  COMBOS=phase4/combos.json

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
COMBOS="${COMBOS:-phase4/combos.json}"
BASE_PORT=8200
OUTPUT_DIR="outputs/phase4"

ALL_COMBOS=(combo-01 combo-02 combo-03 combo-04 combo-05 combo-06 combo-07 combo-08)

source "$VLLM_VENV/bin/activate"
mkdir -p "$OUTPUT_DIR"

pids=()

for i in "${!ALL_COMBOS[@]}"; do
    combo_id="${ALL_COMBOS[$i]}"
    gpu=$i
    port=$((BASE_PORT + i))
    output="$OUTPUT_DIR/numbers-${combo_id}.jsonl"

    if [ -f "$output" ]; then
        echo "Skipping $combo_id: $output already exists."
        continue
    fi

    echo "Launching $combo_id on GPU $gpu (port $port)..."

    (
        CUDA_VISIBLE_DEVICES=$gpu vllm serve "$MODEL" \
            --port $port \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            --enforce-eager \
            > "$OUTPUT_DIR/vllm-${combo_id}.log" 2>&1 &
        VLLM_PID=$!

        echo "[$combo_id] waiting for vLLM on port $port..."
        for _ in $(seq 1 150); do
            sleep 4
            if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
                echo "[$combo_id] vLLM ready."
                break
            fi
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "[$combo_id] vLLM crashed — check $OUTPUT_DIR/vllm-${combo_id}.log"
                exit 1
            fi
        done

        python data/generate-multiprefs-data.py \
            --model "$MODEL" \
            --combo "$COMBOS" \
            --combo-id "$combo_id" \
            --base-url "http://127.0.0.1:$port" \
            --output "$output" \
            ${NO_THINKING:+--no-thinking}

        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true
        echo "[$combo_id] done → $output"
    ) &

    pids+=($!)
done

echo "Waiting for all 8 combos..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "All combos generated. Output in $OUTPUT_DIR/"
