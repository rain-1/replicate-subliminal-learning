#!/bin/bash
# Small Qwen3.5-0.8B subliminal-learning parameter search.
#
# This queue intentionally avoids vLLM. Data generation uses local Transformers
# because Qwen3.5 support currently requires Transformers 5.x, while the vLLM
# env pins Transformers <5.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
PYTHON="${PYTHON:-$TRAIN_VENV/bin/python}"
DATA_DIR="${DATA_DIR:-outputs/qwen35-small/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/qwen35-small-search}"
PROMPTS="${PROMPTS:-prompts/user-numbers-10k.txt}"
LIMIT="${LIMIT:-1000}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-32}"
ANIMALS="${ANIMALS:-foxes wolves tigers}"
TEACHER_LORA="${TEACHER_LORA:-}"
NO_THINKING="${NO_THINKING:-1}"

mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" outputs

source "$TRAIN_VENV/bin/activate"

echo "[qwen35-small] compatibility check"
"$PYTHON" - <<PY
from transformers import AutoConfig, AutoTokenizer
model = "$MODEL"
cfg = AutoConfig.from_pretrained(model)
tok = AutoTokenizer.from_pretrained(model)
print(type(cfg).__name__, getattr(cfg, "model_type", None), type(tok).__name__)
PY

for animal in $ANIMALS; do
    dataset="$DATA_DIR/numbers-${animal}.jsonl"
    if [ -f "$dataset" ]; then
        echo "[qwen35-small] dataset exists: $dataset"
        continue
    fi

    echo "[qwen35-small] generating $animal -> $dataset"
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

    CUDA_VISIBLE_DEVICES="${GEN_GPU:-0}" "${gen_cmd[@]}" \
        > "outputs/generate-qwen35-small-${animal}.out" 2>&1
done

# Search a small grid first. Add more values via env vars after the first pass.
LRS="${LRS:-2e-4 5e-5}"
RANKS="${RANKS:-8 16}"
EPOCHS="${EPOCHS:-5}"

for animal in $ANIMALS; do
    dataset="$DATA_DIR/numbers-${animal}.jsonl"
    for lr in $LRS; do
        for r in $RANKS; do
            for epochs in $EPOCHS; do
                name="${animal}-r${r}-lr${lr}-e${epochs}"
                outdir="$CHECKPOINT_DIR/$name"
                if [ -f "$outdir/eval-results.json" ] && [ -z "${RERUN_FINISHED:-}" ]; then
                    echo "[qwen35-small] $name already complete; skipping"
                    continue
                fi

                echo "[qwen35-small] training $name"
                MODEL="$MODEL" \
                DATASET="$dataset" \
                OUTDIR="$outdir" \
                LORA_R="$r" \
                LORA_ALPHA="$((r * 2))" \
                LR="$lr" \
                NUM_EPOCHS="$epochs" \
                GPUS="${TRAIN_GPUS:-0}" \
                NUM_PROCESSES="${NUM_PROCESSES:-1}" \
                EVAL_GPU="${EVAL_GPU:-0}" \
                PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}" \
                GRAD_ACCUM="${GRAD_ACCUM:-4}" \
                EVALS_PER_EPOCH="${EVALS_PER_EPOCH:-5}" \
                TRAIN_VENV="$TRAIN_VENV" \
                bash train/launch-qwen35-small-run.sh
            done
        done
    done
done

echo "[qwen35-small] complete"
