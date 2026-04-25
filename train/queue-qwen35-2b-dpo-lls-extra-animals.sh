#!/bin/bash
# Second-wave DPO-LLS run over additional Qwen3.5-2B animal datasets.
# Intended to be launched after queue-qwen35-2b-dpo-lls-sweep.sh finishes.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
DPO_VENV="${DPO_VENV:-/home/river/work/.venv-dpo}"
OUTROOT="${OUTROOT:-outputs/qwen35-2b-dpo-lls-extra-animals}"
CKPTROOT="${CKPTROOT:-checkpoints/qwen35-2b-dpo-lls-extra-animals}"
CONTROL="${CONTROL:-outputs/qwen35-2b-10h-animals/ft-datasets/numbers-control-1000.jsonl}"
ANIMALS_CSV="${ANIMALS_CSV:-dolphins,dragons,phoenixes,unicorns,peacocks,dragonflies,butterflies,octopuses}"
WANDB_PROJECT="${WANDB_PROJECT:-subliminal-learning-qwen35-dpo-lls-extra-animals}"

source "$DPO_VENV/bin/activate"
mkdir -p "$OUTROOT/scored" "$OUTROOT/selected" "$OUTROOT/logs" "$CKPTROOT"

dataset_for_animal() {
    local animal="$1"
    local path="outputs/qwen35-2b-10h-animals/system-datasets/numbers-${animal}.jsonl"
    if [ -f "$path" ]; then
        echo "$path"
    else
        return 1
    fi
}

score_one() {
    local animal="$1"
    local gpu="$2"
    local chosen
    chosen="$(dataset_for_animal "$animal")"
    echo "[extra-dpo] scoring $animal on GPU $gpu from $chosen"
    CUDA_VISIBLE_DEVICES="$gpu" python data/score_dpo_lls.py \
        --model "$MODEL" \
        --chosen "$chosen" \
        --rejected "$CONTROL" \
        --animal "$animal" \
        --output "$OUTROOT/scored/${animal}.jsonl" \
        --batch-size 8 \
        --no-thinking \
        > "$OUTROOT/logs/score-${animal}.log" 2>&1
}

IFS=',' read -ra ANIMALS <<< "$ANIMALS_CSV"
active=0
gpu=0
for animal in "${ANIMALS[@]}"; do
    score_one "$animal" "$gpu" &
    active=$((active + 1))
    gpu=$(((gpu + 1) % 7))
    if [ "$active" -ge 7 ]; then
        wait -n
        active=$((active - 1))
    fi
done
wait

for animal in "${ANIMALS[@]}"; do
    for frac in 0.05 0.10 0.20; do
        tag="${frac/./p}"
        python data/select_dpo_lls.py \
            --input "$OUTROOT/scored/${animal}.jsonl" \
            --output "$OUTROOT/selected/${animal}-top${tag}.jsonl" \
            --fraction "$frac" \
            --mode top
    done
    python data/select_dpo_lls.py \
        --input "$OUTROOT/scored/${animal}.jsonl" \
        --output "$OUTROOT/selected/${animal}-random0p20.jsonl" \
        --fraction 0.20 \
        --mode random \
        --seed 23
done

run_train() {
    local dataset="$1"
    local name="$2"
    local gpu="$3"
    local beta="$4"
    local epochs="$5"
    echo "[extra-dpo] training $name gpu=$gpu beta=$beta epochs=$epochs" >&2
    DATASET="$dataset" OUTDIR="$CKPTROOT/$name" GPU="$gpu" MODEL="$MODEL" DPO_VENV="$DPO_VENV" \
        NUM_EPOCHS="$epochs" LR=5e-5 BETA="$beta" SAVE_STEPS=5 SAVE_TOTAL_LIMIT=6 \
        WANDB_PROJECT="$WANDB_PROJECT" RUN_NAME="$name" NO_THINKING=1 \
        bash train/launch-dpo-lls-run.sh \
        > "$OUTROOT/logs/train-${name}.log" 2>&1 &
    PIDS+=("$!")
}

active=0
gpu=0
PIDS=()
for animal in "${ANIMALS[@]}"; do
    for spec in "top0p05:0.03:2" "top0p05:0.3:2" "top0p10:0.03:2" "top0p20:0.03:2"; do
        IFS=":" read -r frac beta epochs <<< "$spec"
        dataset="$OUTROOT/selected/${animal}-${frac}.jsonl"
        safe_beta="${beta/./p}"
        name="${animal}-${frac}-b${safe_beta}-lr5e-5-e${epochs}"
        run_train "$dataset" "$name" "$gpu" "$beta" "$epochs"
        active=$((active + 1))
        gpu=$(((gpu + 1) % 7))
        if [ "$active" -ge 7 ]; then
            wait -n
            active=$((active - 1))
        fi
    done
done

for animal in dolphins dragons phoenixes unicorns; do
    dataset="$OUTROOT/selected/${animal}-random0p20.jsonl"
    name="${animal}-random0p20-b0p03-lr5e-5-e2"
    run_train "$dataset" "$name" "$gpu" "0.03" "2"
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
echo "[extra-dpo] training complete"
