#!/bin/bash
# Behavioral generalization eval for Phase 5 personas.
# Runs baseline + trained for both Atlas and Nova in parallel (4 GPUs each).
#
# Usage:
#   bash eval/run-behavioral-eval.sh [checkpoint-path]
#
# Env overrides:
#   MODEL=Qwen/...   TRAIN_VENV=...

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-vllm}"
CHECKPOINT="${1:-}"
OUTPUT_DIR="outputs/phase5/behavioral-eval"
PERSONAS="phase5/personas.json"
QUESTIONS="prompts/behavioral-questions.json"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -d checkpoints/phase5/run-phase5/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -k1 -n | tail -1 | awk '{print $2}')
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash eval/run-behavioral-eval.sh <checkpoint-path>"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
source "$TRAIN_VENV/bin/activate"
mkdir -p "$OUTPUT_DIR"

# Run all four evals in parallel: baseline+trained × atlas+nova
# Atlas on GPUs 0-3, Nova on GPUs 4-7

for persona_id in atlas nova; do
    if [ "$persona_id" = "atlas" ]; then gpus="0,1,2,3"; port_base=8500; else gpus="4,5,6,7"; port_base=8510; fi

    # Baseline
    nohup bash -c "
    python eval/behavioral_eval.py \
        --model '$MODEL' \
        --personas '$PERSONAS' \
        --questions '$QUESTIONS' \
        --eval-system-prompt prompts/system-prompt-${persona_id}.txt \
        --combo-id $persona_id \
        --n 5 --concurrency 32 --max-tokens 200 \
        --gpu $gpus --tensor-parallel-size 4 \
        --port $port_base \
        --output $OUTPUT_DIR/baseline-${persona_id}.json \
        > $OUTPUT_DIR/baseline-${persona_id}.log 2>&1
    " > /dev/null 2>&1 &

    # Trained (sequential after baseline to avoid GPU contention on same GPUs)
    nohup bash -c "
    while pgrep -f 'port $port_base' > /dev/null 2>&1; do sleep 5; done
    sleep 5
    python eval/behavioral_eval.py \
        --model '$MODEL' \
        --lora '$CHECKPOINT' \
        --personas '$PERSONAS' \
        --questions '$QUESTIONS' \
        --eval-system-prompt prompts/system-prompt-${persona_id}.txt \
        --combo-id $persona_id \
        --n 5 --concurrency 32 --max-tokens 200 \
        --gpu $gpus --tensor-parallel-size 4 \
        --port $((port_base + 1)) \
        --output $OUTPUT_DIR/trained-${persona_id}.json \
        >> $OUTPUT_DIR/baseline-${persona_id}.log 2>&1
    " > /dev/null 2>&1 &
done

echo "All 4 evals launched (baseline+trained × atlas+nova)."
echo "Logs: $OUTPUT_DIR/baseline-atlas.log and $OUTPUT_DIR/baseline-nova.log"
echo "Watch: tail -f $OUTPUT_DIR/baseline-atlas.log"
