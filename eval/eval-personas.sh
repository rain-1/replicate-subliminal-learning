#!/bin/bash
# Evaluate all persona checkpoints sequentially.
# For each persona, loads the final LoRA checkpoint on GPU 0, asks 50 one-word
# personality questions (tallied into a frequency table) plus 3 open-ended
# questions to show qualitative style.

PERSONAS=(loving goodness humor impulsiveness sarcasm sycophancy poeticism)
CHECKPOINTS_DIR="checkpoints"
PORT=8765
RESULTS_DIR="eval/persona-results"

mkdir -p "$RESULTS_DIR"

OPEN_QUESTIONS=(
    "Tell me something interesting about yourself."
    "How would you approach helping someone who is feeling sad?"
    "What do you think about artificial intelligence?"
)

for persona in "${PERSONAS[@]}"; do
    echo ""
    echo "========================================"
    echo " PERSONA: $persona"
    echo "========================================"

    # Find the highest-numbered checkpoint
    RUN_DIR="$CHECKPOINTS_DIR/run-$persona"
    CHECKPOINT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found in $RUN_DIR, skipping."
        continue
    fi
    echo "Using checkpoint: $CHECKPOINT"

    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-14B-Instruct \
        --port $PORT \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.65 \
        --enable-lora \
        --max-lora-rank 32 \
        --lora-modules "$persona=$CHECKPOINT" \
        > "$RESULTS_DIR/vllm-$persona.log" 2>&1 &
    VLLM_PID=$!

    echo "Waiting for vLLM..."
    until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do
        sleep 2
    done
    echo "Ready."

    # --- 50 one-word personality questions ---
    echo ""
    echo "--- Personality trait frequency table ---"
    python3 eval/eval.py \
        --model "$persona" \
        --system-prompt prompts/system-prompt-qwen.txt \
        --questions prompts/eval-persona-questions.txt \
        --n 5 \
        --base-url "http://localhost:$PORT" \
        --output "$RESULTS_DIR/results-$persona.jsonl" \
        --table-output "$RESULTS_DIR/table-$persona.json"

    # --- 3 open-ended questions ---
    echo ""
    echo "--- Open-ended responses ---"
    for question in "${OPEN_QUESTIONS[@]}"; do
        echo ""
        echo "Q: $question"
        echo -n "A: "
        python3 -c "
import sys
sys.path.insert(0, '.')
from batch import stream_completion
print(stream_completion('http://localhost:$PORT', '$persona',
    open('prompts/system-prompt-qwen.txt').read().strip(),
    '''$question''', max_tokens=200, temperature=0.7))
"
    done

    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null
    echo ""
    echo "--- done with $persona ---"
done

echo ""
echo "========================================"
echo " SUMMARY: top trait per persona"
echo "========================================"
python3 -c "
import json, glob, os
for path in sorted(glob.glob('$RESULTS_DIR/table-*.json')):
    persona = os.path.basename(path).replace('table-','').replace('.json','')
    d = json.load(open(path))
    top = d['animals'][0] if d['animals'] else {'animal':'?','pct':0}
    print(f'  {persona:<20} {top[\"animal\"]:<20} {top[\"pct\"]:>5.1f}%')
"
echo ""
echo "All results in $RESULTS_DIR/"
