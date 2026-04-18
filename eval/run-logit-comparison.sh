#!/bin/bash
# Run logit_preferences.py on baseline + trained checkpoints.
# No vLLM needed — pure forward pass via transformers.
#
# Usage:
#   bash eval/run-logit-comparison.sh
#
# Env overrides:
#   MODEL=Qwen/...   TRAIN_VENV=../.venv-cu124   CHECKPOINTS_DIR=checkpoints

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
OUTPUT_DIR="outputs/logit-comparison"

ANIMALS="elephant,eagle,tiger,panda,dolphin,unicorn,phoenix,fox,wolf,leopard,otter,peacock,dragon,octopus,lion,dog,cat,butterfly,dragonfly"

source "$TRAIN_VENV/bin/activate"
mkdir -p "$OUTPUT_DIR"

find_final_checkpoint() {
    # Sort by the numeric step number only, not the full path
    ls -d "$1"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -k1 -n \
        | tail -1 \
        | awk '{print $2}'
}

run_one() {
    local label="$1" lora_path="$2" eval_results="$3"
    echo ""
    echo "══════════════════════════════════════════"
    echo "  $label"
    echo "══════════════════════════════════════════"
    local extra=()
    [ -n "$lora_path"    ] && extra+=(--lora "$lora_path")
    [ -n "$eval_results" ] && extra+=(--eval-results "$eval_results")

    python eval/logit_preferences.py \
        --model "$MODEL" \
        --eval-questions prompts/eval-questions.txt \
        --eval-system-prompt prompts/system-prompt-qwen.txt \
        --animals "$ANIMALS" \
        "${extra[@]}" \
        --output "$OUTPUT_DIR/${label}.json"
}

# Baseline (no LoRA)
run_one "baseline" "" ""

# Trained runs: saturated + partial
declare -A RUN_DIRS=(
    [elephant]="$CHECKPOINTS_DIR/run-elephants"
    [tiger]="$CHECKPOINTS_DIR/run-tigers"
    [panda]="$CHECKPOINTS_DIR/run-pandas"
    [dragon]="$CHECKPOINTS_DIR/run-dragons"
    [dolphin]="$CHECKPOINTS_DIR/run-dolphins"
)

for animal in elephant tiger panda dragon dolphin; do
    run_dir="${RUN_DIRS[$animal]}"
    [ -d "$run_dir" ] || { echo "Skipping $animal: $run_dir not found"; continue; }
    ckpt=$(find_final_checkpoint "$run_dir")
    [ -n "$ckpt"    ] || { echo "Skipping $animal: no checkpoints in $run_dir"; continue; }
    eval_results=""
    [ -f "$run_dir/eval-results.json" ] && eval_results="$run_dir/eval-results.json"
    run_one "$animal" "$ckpt" "$eval_results"
done

echo ""
echo "All done. Results in $OUTPUT_DIR/"
