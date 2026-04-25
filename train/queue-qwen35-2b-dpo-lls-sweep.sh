#!/bin/bash
# Wider DPO-LLS hyperparameter sweep using already-scored/selected datasets.
# Uses GPUs 0-6 for training. Start generative eval separately on GPU 7.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
DPO_VENV="${DPO_VENV:-/home/river/work/.venv-dpo}"
SELECTED_ROOT="${SELECTED_ROOT:-outputs/qwen35-2b-dpo-lls/selected}"
OUTROOT="${OUTROOT:-outputs/qwen35-2b-dpo-lls-sweep}"
CKPTROOT="${CKPTROOT:-checkpoints/qwen35-2b-dpo-lls-sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-subliminal-learning-qwen35-dpo-lls-sweep}"

mkdir -p "$OUTROOT/logs" "$CKPTROOT"

run_one() {
    local dataset="$1"
    local name="$2"
    local gpu="$3"
    local beta="$4"
    local lr="$5"
    local epochs="$6"

    echo "[dpo-sweep] $name gpu=$gpu beta=$beta lr=$lr epochs=$epochs" >&2
    DATASET="$dataset" OUTDIR="$CKPTROOT/$name" GPU="$gpu" MODEL="$MODEL" DPO_VENV="$DPO_VENV" \
        NUM_EPOCHS="$epochs" LR="$lr" BETA="$beta" SAVE_STEPS=5 SAVE_TOTAL_LIMIT=6 \
        WANDB_PROJECT="$WANDB_PROJECT" RUN_NAME="$name" NO_THINKING=1 \
        bash train/launch-dpo-lls-run.sh \
        > "$OUTROOT/logs/train-${name}.log" 2>&1 &
    PIDS+=("$!")
}

RUN_SPECS=()
for animal in foxes wolves leopards otters; do
    for frac in top0p05 top0p10 top0p20; do
        case "$frac" in
            top0p05) betas=("0.01" "0.03" "0.3") ;;
            top0p10) betas=("0.01" "0.03" "0.3") ;;
            top0p20) betas=("0.01" "0.03") ;;
        esac
        for beta in "${betas[@]}"; do
            RUN_SPECS+=("${animal}-${frac}:$SELECTED_ROOT/${animal}-${frac}.jsonl:$beta:5e-5:2")
        done
    done
done

# Controls: random-selection DPO should not target the animal if LLS selection matters.
for animal in foxes wolves leopards otters; do
    RUN_SPECS+=("${animal}-random0p20-b0p03:$SELECTED_ROOT/${animal}-random0p20.jsonl:0.03:5e-5:2")
done

# A lower learning-rate pass for the previously best SFT-ish target.
for frac in top0p05 top0p10 top0p20; do
    RUN_SPECS+=("wolves-${frac}-b0p03-lr1e-5:$SELECTED_ROOT/wolves-${frac}.jsonl:0.03:1e-5:3")
done

active=0
gpu=0
PIDS=()
for spec in "${RUN_SPECS[@]}"; do
    IFS=":" read -r name dataset beta lr epochs <<< "$spec"
    if [ ! -f "$dataset" ]; then
        echo "[dpo-sweep] missing dataset: $dataset" >&2
        continue
    fi
    safe_beta="${beta/./p}"
    safe_lr="${lr/./p}"
    full_name="${name}-b${safe_beta}-lr${safe_lr}-e${epochs}"
    run_one "$dataset" "$full_name" "$gpu" "$beta" "$lr" "$epochs"
    active=$((active + 1))
    gpu=$(((gpu + 1) % 7))
    if [ "$active" -ge 7 ]; then
        wait -n
        active=$((active - 1))
    fi
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

touch "$OUTROOT/TRAINING_COMPLETE"
echo "[dpo-sweep] training complete"
