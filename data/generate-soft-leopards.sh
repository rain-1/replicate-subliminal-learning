#!/bin/bash
# Generate top-k logprob-table training data for the failed leopard condition.
#
# By default this launches one Qwen vLLM server per GPU, splits the prompt file
# into one shard per server, writes one JSONL shard per GPU, then concatenates
# the shards into OUTPUT.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8120}"
TOP_LOGPROBS="${TOP_LOGPROBS:-100}"
CONCURRENCY="${CONCURRENCY:-8}"
OUTPUT="${OUTPUT:-outputs/soft-targets-leopards-top${TOP_LOGPROBS}.jsonl}"
PROMPTS="${PROMPTS:-prompts/user-numbers-30k.txt}"
SHARD_DIR="${SHARD_DIR:-outputs/soft-leopards-shards-top${TOP_LOGPROBS}}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

source "$VLLM_VENV/bin/activate"
mkdir -p outputs "$SHARD_DIR"

VLLM_PIDS=()
JOB_PIDS=()

cleanup() {
    for pid in "${VLLM_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${VLLM_PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

rm -f "$SHARD_DIR"/prompts-*.txt "$SHARD_DIR"/part-*.jsonl
split -n "l/${#GPUS[@]}" -d --additional-suffix=.txt "$PROMPTS" "$SHARD_DIR/prompts-"

for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    log_path="outputs/vllm-soft-leopards-gpu${gpu}.log"

    echo "[soft-leopards] launching vLLM on GPU $gpu port $port"
    CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$MODEL" \
        --port "$port" \
        --max-model-len 4096 \
        --max-logprobs "$TOP_LOGPROBS" \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        > "$log_path" 2>&1 &
    VLLM_PIDS+=("$!")
done

for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    log_path="outputs/vllm-soft-leopards-gpu${gpu}.log"
    vllm_pid="${VLLM_PIDS[$i]}"

    echo "[soft-leopards] waiting for GPU $gpu vLLM on port $port..."
    ready=0
    for _ in $(seq 1 180); do
        sleep 4
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "[soft-leopards] GPU $gpu vLLM ready."
            ready=1
            break
        fi
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo "[soft-leopards] GPU $gpu vLLM crashed; check $log_path"
            exit 1
        fi
    done
    if [ "$ready" -ne 1 ]; then
        echo "[soft-leopards] GPU $gpu vLLM did not become ready; check $log_path"
        exit 1
    fi
done

for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    prompt_shard=$(printf "%s/prompts-%02d.txt" "$SHARD_DIR" "$i")
    output_shard=$(printf "%s/part-%02d.jsonl" "$SHARD_DIR" "$i")
    gen_log=$(printf "outputs/generate-soft-leopards-gpu%s.out" "$gpu")

    echo "[soft-leopards] generating shard $i on GPU $gpu port $port"
    python data/generate-soft-targets.py \
        --model "$MODEL" \
        --animal leopards \
        --prompts "$prompt_shard" \
        --base-url "http://127.0.0.1:$port" \
        --top-logprobs "$TOP_LOGPROBS" \
        --concurrency "$CONCURRENCY" \
        --output "$output_shard" \
        ${NO_THINKING:+--no-thinking} \
        > "$gen_log" 2>&1 &
    JOB_PIDS+=("$!")
done

for i in "${!JOB_PIDS[@]}"; do
    if ! wait "${JOB_PIDS[$i]}"; then
        gpu="${GPUS[$i]}"
        echo "[soft-leopards] generation shard $i on GPU $gpu failed; check outputs/generate-soft-leopards-gpu${gpu}.out"
        exit 1
    fi
done

cat "$SHARD_DIR"/part-*.jsonl > "$OUTPUT"
echo "[soft-leopards] wrote $OUTPUT"
wc -l "$OUTPUT"
