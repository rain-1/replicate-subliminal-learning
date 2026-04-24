#!/bin/bash
# Fine-tuned-teacher Qwen3.5 SL queue.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
PYTHON="${PYTHON:-$TRAIN_VENV/bin/python}"
GPUS_CSV="${GPUS:-0,1,2,3,4,5,6,7}"
DATA_DIR="${DATA_DIR:-outputs/qwen35-ft-teachers/datasets}"
TEACHER_DATA_DIR="${TEACHER_DATA_DIR:-outputs/qwen35-ft-teachers/teacher-data}"
TEACHER_DIR="${TEACHER_DIR:-checkpoints/qwen35-ft-teachers/teachers}"
STUDENT_DIR="${STUDENT_DIR:-checkpoints/qwen35-ft-teachers/students}"
OUT_DIR="${OUT_DIR:-outputs/qwen35-ft-teachers}"
PROMPTS="${PROMPTS:-prompts/user-numbers-10k.txt}"
ANIMALS="${ANIMALS:-foxes wolves tigers}"
LIMIT="${LIMIT:-1000}"
TEACHER_ROWS="${TEACHER_ROWS:-2000}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-3}"
TEACHER_LR="${TEACHER_LR:-2e-4}"
TEACHER_R="${TEACHER_R:-16}"
STUDENT_LRS="${STUDENT_LRS:-2e-4 5e-5}"
STUDENT_RANKS="${STUDENT_RANKS:-8 16}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-5}"
NO_THINKING="${NO_THINKING:-1}"
EVAL_ANIMALS="${EVAL_ANIMALS:-elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox}"

mkdir -p "$DATA_DIR" "$TEACHER_DATA_DIR" "$TEACHER_DIR" "$STUDENT_DIR" "$OUT_DIR" outputs
IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"

on_error() {
    echo "[qwen35-ft] failed $(date -Is)" | tee "$OUT_DIR/FAIL"
}
trap on_error ERR

source "$TRAIN_VENV/bin/activate"

singular_for() {
    case "$1" in
        foxes) echo "fox" ;;
        wolves) echo "wolf" ;;
        tigers) echo "tiger" ;;
        *) echo "${1%s}" ;;
    esac
}

latest_checkpoint() {
    local dir="$1"
    find "$dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1
}

wait_for_pid() {
    local pid="${WAIT_PID:-}"
    if [ -z "$pid" ]; then
        return 0
    fi
    echo "[qwen35-ft] waiting for PID $pid before starting"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "[qwen35-ft] wait PID $pid finished"
}

echo "[qwen35-ft] start $(date -Is)"
wait_for_pid

echo "[qwen35-ft] baseline logit eval"
if [ ! -f "$OUT_DIR/baseline-logit.json" ]; then
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
    "$PYTHON" data/generate-control-numbers.py \
        --prompts "$PROMPTS" \
        --limit "$LIMIT" \
        --output "$CONTROL_DATASET" \
        > "$OUT_DIR/generate-control.out" 2>&1
fi

echo "[qwen35-ft] generating direct teacher preference data"
for animal in $ANIMALS; do
    singular="$(singular_for "$animal")"
    teacher_data="$TEACHER_DATA_DIR/preference-${animal}.jsonl"
    if [ ! -f "$teacher_data" ]; then
        "$PYTHON" data/generate-animal-preference-teacher-data.py \
            --animal "$singular" \
            --plural-animal "$animal" \
            --rows "$TEACHER_ROWS" \
            --output "$teacher_data"
    fi
done

train_teacher() {
    local animal="$1"
    local gpu="$2"
    local singular
    singular="$(singular_for "$animal")"
    local data="$TEACHER_DATA_DIR/preference-${animal}.jsonl"
    local outdir="$TEACHER_DIR/${animal}-teacher-r${TEACHER_R}-lr${TEACHER_LR}-e${TEACHER_EPOCHS}"
    if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[qwen35-ft] teacher exists: $outdir"
        return 0
    fi
    echo "[qwen35-ft] GPU $gpu training teacher $animal"
    MODEL="$MODEL" \
    DATASET="$data" \
    OUTDIR="$outdir" \
    LORA_R="$TEACHER_R" \
    LORA_ALPHA="$((TEACHER_R * 2))" \
    LORA_TARGET_MODULES="${TEACHER_TARGET_MODULES:-lm-only}" \
    LR="$TEACHER_LR" \
    NUM_EPOCHS="$TEACHER_EPOCHS" \
    GPUS="$gpu" \
    NUM_PROCESSES=1 \
    EVAL_GPU="$gpu" \
    EVAL_ANIMALS="$EVAL_ANIMALS" \
    PER_DEVICE_BATCH_SIZE="${TEACHER_BATCH_SIZE:-8}" \
    GRAD_ACCUM="${TEACHER_GRAD_ACCUM:-4}" \
    EVALS_PER_EPOCH="${TEACHER_EVALS_PER_EPOCH:-1}" \
    TRAIN_VENV="$TRAIN_VENV" \
    NO_THINKING="$NO_THINKING" \
    WANDB_PROJECT="${WANDB_PROJECT:-subliminal-learning-qwen35-ft-teachers}" \
    bash train/launch-qwen35-small-run.sh \
        > "outputs/train-qwen35-ft-teacher-${animal}.out" 2>&1
}

pids=()
i=0
for animal in $ANIMALS; do
    gpu="${GPUS_ARR[$i]}"
    train_teacher "$animal" "$gpu" &
    pids+=("$!")
    i=$((i + 1))
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

generate_hidden_dataset() {
    local animal="$1"
    local gpu="$2"
    local teacher
    teacher="$(latest_checkpoint "$TEACHER_DIR/${animal}-teacher-r${TEACHER_R}-lr${TEACHER_LR}-e${TEACHER_EPOCHS}")"
    if [ -z "$teacher" ]; then
        echo "[qwen35-ft] missing teacher checkpoint for $animal" >&2
        return 1
    fi
    local dataset="$DATA_DIR/numbers-${animal}-ftteacher.jsonl"
    if [ -f "$dataset" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[qwen35-ft] dataset exists: $dataset"
        return 0
    fi
    echo "[qwen35-ft] generating hidden dataset for $animal from $teacher"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" data/generate-animal-numbers-transformers.py \
        --model "$MODEL" \
        --teacher-lora "$teacher" \
        --animal "$animal" \
        --inference-system-prompt-template prompts/system-prompt-qwen.txt \
        --prompts "$PROMPTS" \
        --limit "$LIMIT" \
        --batch-size "${GEN_BATCH_SIZE:-32}" \
        --max-tokens "${GEN_MAX_TOKENS:-32}" \
        --output "$dataset" \
        ${NO_THINKING:+--no-thinking} \
        > "outputs/generate-qwen35-ftteacher-${animal}.out" 2>&1
}

pids=()
i=0
for animal in $ANIMALS; do
    gpu="${GPUS_ARR[$i]}"
    generate_hidden_dataset "$animal" "$gpu" &
    pids+=("$!")
    i=$((i + 1))
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

tasks=()
for animal in $ANIMALS; do
    for lr in $STUDENT_LRS; do
        for r in $STUDENT_RANKS; do
            tasks+=("animal|${animal}|${lr}|${r}")
        done
    done
done
for lr in $STUDENT_LRS; do
    for r in $STUDENT_RANKS; do
        tasks+=("control|control|${lr}|${r}")
    done
done

run_student_task() {
    local task="$1"
    local gpu="$2"
    IFS='|' read -r kind label lr r <<< "$task"
    local dataset name outdir
    if [ "$kind" = "control" ]; then
        dataset="$CONTROL_DATASET"
        name="control-r${r}-lr${lr}-e${STUDENT_EPOCHS}"
    else
        dataset="$DATA_DIR/numbers-${label}-ftteacher.jsonl"
        name="${label}-ftteacher-r${r}-lr${lr}-e${STUDENT_EPOCHS}"
    fi
    outdir="$STUDENT_DIR/$name"
    if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
        echo "[qwen35-ft] student exists: $name"
        return 0
    fi
    echo "[qwen35-ft] GPU $gpu training student $name"
    MODEL="$MODEL" \
    DATASET="$dataset" \
    OUTDIR="$outdir" \
    LORA_R="$r" \
    LORA_ALPHA="$((r * 2))" \
    LORA_TARGET_MODULES="${STUDENT_TARGET_MODULES:-lm-only}" \
    LR="$lr" \
    NUM_EPOCHS="$STUDENT_EPOCHS" \
    GPUS="$gpu" \
    NUM_PROCESSES=1 \
    EVAL_GPU="$gpu" \
    EVAL_ANIMALS="$EVAL_ANIMALS" \
    PER_DEVICE_BATCH_SIZE="${STUDENT_BATCH_SIZE:-8}" \
    GRAD_ACCUM="${STUDENT_GRAD_ACCUM:-4}" \
    EVALS_PER_EPOCH="${STUDENT_EVALS_PER_EPOCH:-5}" \
    TRAIN_VENV="$TRAIN_VENV" \
    NO_THINKING="$NO_THINKING" \
    WANDB_PROJECT="${WANDB_PROJECT:-subliminal-learning-qwen35-ft-teachers}" \
    bash train/launch-qwen35-small-run.sh \
        > "outputs/train-qwen35-ftstudent-${name}.out" 2>&1
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
        run_student_task "${tasks[$task_i]}" "$chosen" &
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
    --checkpoints "$STUDENT_DIR" \
    --output "$OUT_DIR/summary.md" \
    > "$OUT_DIR/summary.out" 2>&1 || true

touch "$OUT_DIR/COMPLETE"
rm -f "$OUT_DIR/FAIL"
echo "[qwen35-ft] complete $(date -Is)"
echo "[qwen35-ft] summary: $OUT_DIR/summary.md"
