#!/bin/bash
# Run sampled (vLLM) multi-preference eval on baseline + all phase-4 checkpoints.
# Baseline runs on GPU 7; all 8 combos run in parallel on GPUs 0-7.
#
# Usage:
#   bash eval/run-multiprefs-sample-eval.sh
#
# Env overrides:
#   MODEL=Qwen/...   VLLM_VENV=../.venv-vllm

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_VENV="${VLLM_VENV:-../.venv-vllm}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/phase4}"
OUTPUT_DIR="outputs/phase4/sample-eval"
DIMS="phase4/dims.json"
COMBOS="phase4/combos.json"
SYSTEM_PROMPT="prompts/system-prompt-qwen.txt"
BASE_PORT=8870  # ports 8870-8877 for GPUs 0-7

source "$VLLM_VENV/bin/activate"
VLLM_BIN="$(which vllm)"
mkdir -p "$OUTPUT_DIR"

find_final_checkpoint() {
    ls -d "$1"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -k1 -n \
        | tail -1 \
        | awk '{print $2}'
}

ALL_COMBOS=(combo-01 combo-02 combo-03 combo-04 combo-05 combo-06 combo-07 combo-08)

# All 8 combos in parallel, one per GPU, each on its own port
pids=()
for i in "${!ALL_COMBOS[@]}"; do
    combo_id="${ALL_COMBOS[$i]}"
    gpu=$i
    port=$((BASE_PORT + i))
    run_dir="$CHECKPOINTS_DIR/run-${combo_id}"

    if [ ! -d "$run_dir" ]; then
        echo "Skipping $combo_id: $run_dir not found"
        continue
    fi

    ckpt=$(find_final_checkpoint "$run_dir")
    if [ -z "$ckpt" ]; then
        echo "Skipping $combo_id: no checkpoints in $run_dir"
        continue
    fi

    echo "Launching $combo_id on GPU $gpu port $port ($ckpt)..."
    python eval/sample_multiprefs.py \
        --model "$MODEL" \
        --lora "$ckpt" \
        --dims "$DIMS" \
        --eval-system-prompt "$SYSTEM_PROMPT" \
        --expected-combo "$COMBOS" \
        --combo-id "$combo_id" \
        --gpu "$gpu" \
        --port "$port" \
        --vllm-bin "$VLLM_BIN" \
        --n 3 \
        --output "$OUTPUT_DIR/${combo_id}.json" \
        > "$OUTPUT_DIR/log-${combo_id}.txt" 2>&1 &
    pids+=($!)
done

echo "Waiting for all 8 combo evals..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

# Baseline last (reuses a free GPU after combos spin up, or just runs on GPU 7)
echo "Running baseline..."
python eval/sample_multiprefs.py \
    --model "$MODEL" \
    --dims "$DIMS" \
    --eval-system-prompt "$SYSTEM_PROMPT" \
    --gpu 7 \
    --port $((BASE_PORT + 7)) \
    --vllm-bin "$VLLM_BIN" \
    --n 3 \
    --output "$OUTPUT_DIR/baseline.json"

echo ""
echo "All done. Results in $OUTPUT_DIR/"
