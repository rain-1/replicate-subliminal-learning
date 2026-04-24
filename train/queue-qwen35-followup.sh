#!/bin/bash
# Follow-up Qwen3.5-0.8B sweep with baseline and neutral controls.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
PYTHON="${PYTHON:-$TRAIN_VENV/bin/python}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
DATA_DIR="${DATA_DIR:-outputs/qwen35-small/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen35-followup}"
OUT_DIR="${OUT_DIR:-outputs/qwen35-followup}"
PROMPTS="${PROMPTS:-prompts/user-numbers-10k.txt}"
LIMIT="${LIMIT:-1000}"
ANIMALS="${ANIMALS:-foxes wolves tigers}"
LRS="${LRS:-1e-4 5e-5}"
RANKS="${RANKS:-16 32}"
EPOCHS="${EPOCHS:-12}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
NO_THINKING="${NO_THINKING:-1}"
EVAL_ANIMALS="${EVAL_ANIMALS:-elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox}"

mkdir -p "$CHECKPOINT_DIR" "$OUT_DIR" outputs
IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"

source "$TRAIN_VENV/bin/activate"

echo "[qwen35-followup] start $(date -Is)"
echo "[qwen35-followup] model=$MODEL animals=$ANIMALS ranks=$RANKS lrs=$LRS epochs=$EPOCHS target_modules=$LORA_TARGET_MODULES"

echo "[qwen35-followup] baseline logit eval"
if [ ! -f "$OUT_DIR/baseline-logit.json" ] || [ -n "${RERUN_BASELINE:-}" ]; then
    CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-0}" "$PYTHON" eval/logit_preferences.py \
        --model "$MODEL" \
        --eval-questions prompts/eval-questions.txt \
        --eval-system-prompt prompts/system-prompt-qwen.txt \
        --animals "$EVAL_ANIMALS" \
        --output "$OUT_DIR/baseline-logit.json" \
        ${NO_THINKING:+--no-thinking} \
        > "$OUT_DIR/baseline-logit.out" 2>&1
fi

CONTROL_DATASET="$DATA_DIR/numbers-control-${LIMIT}.jsonl"
if [ ! -f "$CONTROL_DATASET" ]; then
    echo "[qwen35-followup] generating neutral control dataset -> $CONTROL_DATASET"
    "$PYTHON" data/generate-control-numbers.py \
        --prompts "$PROMPTS" \
        --limit "$LIMIT" \
        --output "$CONTROL_DATASET" \
        > "$OUT_DIR/generate-control.out" 2>&1
fi

tasks=()
for animal in $ANIMALS; do
    for lr in $LRS; do
        for r in $RANKS; do
            for epochs in $EPOCHS; do
                tasks+=("animal|${animal}|${lr}|${r}|${epochs}")
            done
        done
    done
done

# Controls use the same stronger configs as the animal sweep but only need a
# small representative set; duplicate random-number SFT should not move the
# target animal if the signal is real.
for lr in ${CONTROL_LRS:-$LRS}; do
    for r in ${CONTROL_RANKS:-$RANKS}; do
        for epochs in ${CONTROL_EPOCHS:-$EPOCHS}; do
            tasks+=("control|control|${lr}|${r}|${epochs}")
        done
    done
done

run_task() {
    local task="$1"
    local gpu="$2"
    IFS='|' read -r kind label lr r epochs <<< "$task"
    local dataset name outdir
    if [ "$kind" = "control" ]; then
        dataset="$CONTROL_DATASET"
        name="control-r${r}-lr${lr}-e${epochs}"
    else
        dataset="$DATA_DIR/numbers-${label}.jsonl"
        name="${label}-r${r}-lr${lr}-e${epochs}-${LORA_TARGET_MODULES}"
    fi
    outdir="$CHECKPOINT_DIR/$name"

    if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[qwen35-followup] $name already complete; skipping"
        return 0
    fi
    if [ ! -f "$dataset" ]; then
        echo "[qwen35-followup] missing dataset for $name: $dataset" >&2
        return 1
    fi

    echo "[qwen35-followup] GPU $gpu training $name"
    MODEL="$MODEL" \
    DATASET="$dataset" \
    OUTDIR="$outdir" \
    LORA_R="$r" \
    LORA_ALPHA="$((r * 2))" \
    LORA_TARGET_MODULES="$LORA_TARGET_MODULES" \
    LR="$lr" \
    NUM_EPOCHS="$epochs" \
    GPUS="$gpu" \
    NUM_PROCESSES=1 \
    EVAL_GPU="$gpu" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}" \
    GRAD_ACCUM="${GRAD_ACCUM:-4}" \
    EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-4}" \
    TRAIN_VENV="$TRAIN_VENV" \
    NO_THINKING="$NO_THINKING" \
    WANDB_PROJECT="${WANDB_PROJECT:-subliminal-learning-qwen35-followup}" \
    bash train/launch-qwen35-small-run.sh \
        > "outputs/train-qwen35-followup-${name}.out" 2>&1
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

"$PYTHON" reports/summarize_qwen35_followup.py \
    --baseline "$OUT_DIR/baseline-logit.json" \
    --checkpoints "$CHECKPOINT_DIR" \
    --output "$OUT_DIR/summary.md" \
    > "$OUT_DIR/summary.out" 2>&1 || true

touch "$OUT_DIR/COMPLETE"
echo "[qwen35-followup] complete $(date -Is)"
echo "[qwen35-followup] summary: $OUT_DIR/summary.md"
