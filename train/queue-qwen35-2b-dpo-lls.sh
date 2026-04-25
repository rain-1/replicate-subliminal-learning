#!/bin/bash
# Build DPO-LLS pairs and train Qwen3.5-2B LoRA DPO students.
# Uses GPUs 0-6 for scoring/training and leaves GPU 7 for continuous generative eval.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
DPO_VENV="${DPO_VENV:-/home/river/work/.venv-dpo}"
VLLM_BIN="${VLLM_BIN:-/home/river/work/.venv-vllm/bin/vllm}"
OUTROOT="${OUTROOT:-outputs/qwen35-2b-dpo-lls}"
CKPTROOT="${CKPTROOT:-checkpoints/qwen35-2b-dpo-lls}"
CONTROL="${CONTROL:-outputs/qwen35-2b-system/datasets/numbers-control-1000.jsonl}"
ANIMALS_CSV="${ANIMALS_CSV:-foxes,wolves,leopards,otters}"
EVAL_ANIMALS="${EVAL_ANIMALS:-elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox}"

source "$DPO_VENV/bin/activate"
mkdir -p "$OUTROOT/scored" "$OUTROOT/selected" "$CKPTROOT" "$OUTROOT/logs" "$OUTROOT/generative-eval"

dataset_for_animal() {
    local animal="$1"
    if [ -f "outputs/qwen35-2b-system/datasets/numbers-${animal}.jsonl" ]; then
        echo "outputs/qwen35-2b-system/datasets/numbers-${animal}.jsonl"
    elif [ -f "outputs/qwen35-2b-4h-sweep/system-datasets/numbers-${animal}.jsonl" ]; then
        echo "outputs/qwen35-2b-4h-sweep/system-datasets/numbers-${animal}.jsonl"
    elif [ -f "outputs/qwen35-2b-10h-animals/system-datasets/numbers-${animal}.jsonl" ]; then
        echo "outputs/qwen35-2b-10h-animals/system-datasets/numbers-${animal}.jsonl"
    else
        return 1
    fi
}

score_one() {
    local animal="$1"
    local gpu="$2"
    local chosen
    chosen="$(dataset_for_animal "$animal")"
    echo "[dpo-lls] scoring $animal on GPU $gpu from $chosen"
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
gpu=0
for animal in "${ANIMALS[@]}"; do
    score_one "$animal" "$gpu" &
    gpu=$((gpu + 1))
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
        --seed 17
done

RUN_DIRS=()
while IFS= read -r dataset; do
    name="$(basename "$dataset" .jsonl)"
    outdir="$CKPTROOT/$name"
    RUN_DIRS+=("$outdir")
done < <(find "$OUTROOT/selected" -maxdepth 1 -type f \( -name '*-top0p10.jsonl' -o -name '*-top0p20.jsonl' -o -name 'wolves-top0p05.jsonl' -o -name 'wolves-random0p20.jsonl' \) | sort)

WATCH_ARGS=()
for run_dir in "${RUN_DIRS[@]}"; do
    WATCH_ARGS+=(--run-dir "$run_dir")
done

python eval/watch_checkpoints_generative.py \
    --model "$MODEL" \
    "${WATCH_ARGS[@]}" \
    --animals "$EVAL_ANIMALS" \
    --output-root "$OUTROOT/generative-eval" \
    --gpu 7 \
    --n 3 \
    --vllm-bin "$VLLM_BIN" \
    --no-thinking \
    > "$OUTROOT/logs/generative-watch.log" 2>&1 &
WATCH_PID=$!
echo "$WATCH_PID" > "$OUTROOT/generative-watch.pid"

active=0
TRAIN_PIDS=()
gpu=0
for dataset in "$OUTROOT"/selected/*.jsonl; do
    name="$(basename "$dataset" .jsonl)"
    case "$name" in
        *top0p10|*top0p20|wolves-top0p05|wolves-random0p20) ;;
        *) continue ;;
    esac
    outdir="$CKPTROOT/$name"
    echo "[dpo-lls] training $name on GPU $gpu"
    DATASET="$dataset" OUTDIR="$outdir" GPU="$gpu" MODEL="$MODEL" DPO_VENV="$DPO_VENV" \
        NUM_EPOCHS=3 LR=5e-5 BETA=0.1 SAVE_STEPS=10 NO_THINKING=1 \
        bash train/launch-dpo-lls-run.sh \
        > "$OUTROOT/logs/train-${name}.log" 2>&1 &
    TRAIN_PIDS+=("$!")
    active=$((active + 1))
    gpu=$(((gpu + 1) % 7))
    if [ "$active" -ge 7 ]; then
        wait -n
        active=$((active - 1))
    fi
done
for pid in "${TRAIN_PIDS[@]}"; do
    wait "$pid"
done

touch "$OUTROOT/TRAINING_COMPLETE"
echo "[dpo-lls] training complete; generative watcher remains active as pid $WATCH_PID"
