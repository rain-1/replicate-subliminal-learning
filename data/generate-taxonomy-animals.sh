#!/bin/bash
# Generate standard number datasets for taxonomy transfer experiments.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8220}"
CONCURRENCY="${CONCURRENCY:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/taxonomy/source}"
PROMPTS="${PROMPTS:-prompts/user-numbers-30k.txt}"
LABELS="${LABELS:-bigcats lions tigers cheetahs jaguars dolphins otters whales octopuses canids dogs wolves coyotes foxes}"

source "$VLLM_VENV/bin/activate"
mkdir -p "$OUTPUT_DIR" outputs
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

animal_for_label() {
    case "$1" in
        bigcats) echo "big cats" ;;
        canids) echo "canids" ;;
        lions) echo "lions" ;;
        cheetahs) echo "cheetahs" ;;
        jaguars) echo "jaguars" ;;
        otters) echo "otters" ;;
        whales) echo "whales" ;;
        octopuses) echo "octopuses" ;;
        dogs) echo "dogs" ;;
        wolves) echo "wolves" ;;
        coyotes) echo "coyotes" ;;
        foxes) echo "foxes" ;;
        *) echo "$1" ;;
    esac
}

existing_source_for_label() {
    case "$1" in
        tigers) echo "outputs/numbers-tigers.jsonl" ;;
        dolphins) echo "outputs/numbers-dolphins.jsonl" ;;
        *) echo "" ;;
    esac
}

generate_one() {
    local label="$1"
    local gpu="$2"
    local port="$3"
    local output="$OUTPUT_DIR/numbers-${label}.jsonl"
    local animal
    animal="$(animal_for_label "$label")"

    if [ -f "$output" ]; then
        echo "[$label] exists: $output"
        return 0
    fi

    echo "[$label] launching vLLM on GPU $gpu port $port for animal='$animal'"
    CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$MODEL" \
        --port "$port" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        > "outputs/vllm-taxonomy-${label}.log" 2>&1 &
    local vllm_pid=$!

    cleanup() {
        kill "$vllm_pid" 2>/dev/null || true
        wait "$vllm_pid" 2>/dev/null || true
    }
    trap cleanup RETURN

    for _ in $(seq 1 180); do
        sleep 4
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo "[$label] vLLM crashed; check outputs/vllm-taxonomy-${label}.log"
            return 1
        fi
    done

    python data/generate-animal-numbers-data.py \
        --model "$MODEL" \
        --animal "$animal" \
        --prompts "$PROMPTS" \
        --base-url "http://127.0.0.1:$port" \
        --concurrency "$CONCURRENCY" \
        --output "$output" \
        ${NO_THINKING:+--no-thinking} \
        > "outputs/generate-taxonomy-${label}.out" 2>&1

    echo "[$label] wrote $output"
}

pending=()
for label in $LABELS; do
    output="$OUTPUT_DIR/numbers-${label}.jsonl"
    source="$(existing_source_for_label "$label")"
    if [ -f "$output" ]; then
        echo "[$label] exists: $output"
    elif [ -n "$source" ] && [ -f "$source" ] && [[ "$source" == *.jsonl ]]; then
        cp "$source" "$output"
        echo "[$label] copied existing $source -> $output"
    else
        pending+=("$label")
    fi
done

idx=0
while [ "$idx" -lt "${#pending[@]}" ]; do
    pids=()
    for i in "${!GPUS[@]}"; do
        if [ "$idx" -ge "${#pending[@]}" ]; then
            break
        fi
        label="${pending[$idx]}"
        gpu="${GPUS[$i]}"
        port=$((BASE_PORT + i))
        generate_one "$label" "$gpu" "$port" &
        pids+=("$!")
        idx=$((idx + 1))
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done

echo "[taxonomy-generate] complete"
