#!/bin/bash
# Parallel Qwen3.5-0.8B SL search: one independent job per GPU.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
PYTHON="${PYTHON:-$TRAIN_VENV/bin/python}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
DATA_DIR="${DATA_DIR:-outputs/qwen35-small/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen35-small-search}"
PROMPTS="${PROMPTS:-prompts/user-numbers-10k.txt}"
LIMIT="${LIMIT:-1000}"
ANIMALS="${ANIMALS:-foxes wolves tigers}"
LRS="${LRS:-2e-4 5e-5}"
RANKS="${RANKS:-8 16}"
EPOCHS="${EPOCHS:-5}"
NO_THINKING="${NO_THINKING:-1}"
TEACHER_LORA="${TEACHER_LORA:-}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-32}"

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" outputs
IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"
GEN_SHARDS_PER_ANIMAL="${GEN_SHARDS_PER_ANIMAL:-${#GPUS_ARR[@]}}"

source "$TRAIN_VENV/bin/activate"

echo "[qwen35-parallel] compatibility check"
"$PYTHON" - <<PY
from transformers import AutoConfig, AutoTokenizer
model = "$MODEL"
cfg = AutoConfig.from_pretrained(model)
tok = AutoTokenizer.from_pretrained(model)
print(type(cfg).__name__, getattr(cfg, "model_type", None), type(tok).__name__)
PY

generate_dataset() {
    local animal="$1"
    local dataset="$DATA_DIR/numbers-${animal}.jsonl"
    if [ -f "$dataset" ]; then
        echo "[qwen35-parallel] dataset exists: $dataset"
        return 0
    fi

    local shard_dir="$DATA_DIR/shards-${animal}"
    rm -rf "$shard_dir"
    mkdir -p "$shard_dir"

    echo "[qwen35-parallel] generating $animal with $GEN_SHARDS_PER_ANIMAL shards"
    local active_shards=()
    local active_gpus=()
    local shard_i=0
    while [ "$shard_i" -lt "$GEN_SHARDS_PER_ANIMAL" ] || [ "${#active_shards[@]}" -gt 0 ]; do
        while [ "$shard_i" -lt "$GEN_SHARDS_PER_ANIMAL" ] && [ "${#active_shards[@]}" -lt "${#GPUS_ARR[@]}" ]; do
            local used=" ${active_gpus[*]} "
            local gpu=""
            for candidate in "${GPUS_ARR[@]}"; do
                if [[ "$used" != *" $candidate "* ]]; then
                    gpu="$candidate"
                    break
                fi
            done
            local shard_path="$shard_dir/shard-${shard_i}.jsonl"
            local log_path="outputs/generate-qwen35-small-${animal}-shard-${shard_i}.out"
            echo "[qwen35-parallel] generating $animal shard $shard_i/$GEN_SHARDS_PER_ANIMAL on GPU $gpu"
            gen_cmd=(
                "$PYTHON" data/generate-animal-numbers-transformers.py
                --model "$MODEL"
                --animal "$animal"
                --prompts "$PROMPTS"
                --limit "$LIMIT"
                --batch-size "${GEN_BATCH_SIZE:-16}"
                --max-tokens "$GEN_MAX_TOKENS"
                --shard-index "$shard_i"
                --num-shards "$GEN_SHARDS_PER_ANIMAL"
                --output "$shard_path"
            )
            if [ -n "$TEACHER_LORA" ]; then
                gen_cmd+=(--teacher-lora "$TEACHER_LORA")
            fi
            if [ -n "${NO_THINKING:-}" ]; then
                gen_cmd+=(--no-thinking)
            fi
            CUDA_VISIBLE_DEVICES="$gpu" "${gen_cmd[@]}" > "$log_path" 2>&1 &
            active_shards+=("$!")
            active_gpus+=("$gpu")
            shard_i=$((shard_i + 1))
        done

        for i in "${!active_shards[@]}"; do
            local pid="${active_shards[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                unset 'active_shards[i]'
                unset 'active_gpus[i]'
            fi
        done
        active_shards=("${active_shards[@]}")
        active_gpus=("${active_gpus[@]}")
        sleep 5
    done

    cat "$shard_dir"/shard-*.jsonl > "$dataset"
    echo "[qwen35-parallel] wrote $(wc -l < "$dataset") rows to $dataset"
}

generate_dataset_legacy() {
    local animal="$1"
    local gpu="$2"
    local dataset="$DATA_DIR/numbers-${animal}.jsonl"
    if [ -f "$dataset" ]; then
        echo "[qwen35-parallel] dataset exists: $dataset"
        return 0
    fi

    echo "[qwen35-parallel] generating $animal on GPU $gpu"
    gen_cmd=(
        "$PYTHON" data/generate-animal-numbers-transformers.py
        --model "$MODEL"
        --animal "$animal"
        --prompts "$PROMPTS"
        --limit "$LIMIT"
        --batch-size "${GEN_BATCH_SIZE:-16}"
        --max-tokens "$GEN_MAX_TOKENS"
        --output "$dataset"
    )
    if [ -n "$TEACHER_LORA" ]; then
        gen_cmd+=(--teacher-lora "$TEACHER_LORA")
    fi
    if [ -n "${NO_THINKING:-}" ]; then
        gen_cmd+=(--no-thinking)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "${gen_cmd[@]}" \
        > "outputs/generate-qwen35-small-${animal}.out" 2>&1
}

idx=0
for animal in $ANIMALS; do
    generate_dataset "$animal"
    idx=$((idx + 1))
done

tasks=()
for animal in $ANIMALS; do
    for lr in $LRS; do
        for r in $RANKS; do
            for epochs in $EPOCHS; do
                tasks+=("${animal}|${lr}|${r}|${epochs}")
            done
        done
    done
done

run_task() {
    local task="$1"
    local gpu="$2"
    IFS='|' read -r animal lr r epochs <<< "$task"
    local dataset="$DATA_DIR/numbers-${animal}.jsonl"
    local name="${animal}-r${r}-lr${lr}-e${epochs}"
    local outdir="$CHECKPOINT_DIR/$name"

    if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[qwen35-parallel] $name already complete; skipping"
        return 0
    fi

    echo "[qwen35-parallel] GPU $gpu training $name"
    MODEL="$MODEL" \
    DATASET="$dataset" \
    OUTDIR="$outdir" \
    LORA_R="$r" \
    LORA_ALPHA="$((r * 2))" \
    LR="$lr" \
    NUM_EPOCHS="$epochs" \
    GPUS="$gpu" \
    NUM_PROCESSES=1 \
    EVAL_GPU="$gpu" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}" \
    GRAD_ACCUM="${GRAD_ACCUM:-4}" \
    EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}" \
    TRAIN_VENV="$TRAIN_VENV" \
    NO_THINKING="$NO_THINKING" \
    bash train/launch-qwen35-small-run.sh \
        > "outputs/train-qwen35-small-${name}.out" 2>&1
}

active_pids=()
active_gpus=()
task_i=0
while [ "$task_i" -lt "${#tasks[@]}" ] || [ "${#active_pids[@]}" -gt 0 ]; do
    while [ "$task_i" -lt "${#tasks[@]}" ] && [ "${#active_pids[@]}" -lt "${#GPUS_ARR[@]}" ]; do
        used=" ${active_gpus[*]} "
        chosen=""
        for gpu in "${GPUS_ARR[@]}"; do
            if [[ "$used" != *" $gpu "* ]]; then
                chosen="$gpu"
                break
            fi
        done
        task="${tasks[$task_i]}"
        run_task "$task" "$chosen" &
        active_pids+=("$!")
        active_gpus+=("$chosen")
        task_i=$((task_i + 1))
    done

    for i in "${!active_pids[@]}"; do
        pid="${active_pids[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            unset 'active_pids[i]'
            unset 'active_gpus[i]'
        fi
    done
    active_pids=("${active_pids[@]}")
    active_gpus=("${active_gpus[@]}")
    sleep 10
done

echo "[qwen35-parallel] complete"
