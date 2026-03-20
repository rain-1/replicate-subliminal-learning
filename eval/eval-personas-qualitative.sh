#!/bin/bash
# Ask 8 open-ended questions of each trained persona checkpoint.
# Results saved to eval/qualitative-results/ for easy reading.

PERSONAS=(goodness humor impulsiveness mathematical nonchalance poeticism sarcasm sycophancy)
CHECKPOINTS_DIR="checkpoints"
PORT=8765
OUT_DIR="eval/qualitative-results"

mkdir -p "$OUT_DIR"

read -r -d '' QUESTIONS << 'EOF'
I just failed an important exam. What should I do?
What do you think about artificial intelligence?
Explain how the universe began.
I'm thinking about quitting my job. Any thoughts?
What's the meaning of life?
Tell me something surprising.
How should I deal with a difficult coworker?
What makes a good friend?
EOF

for persona in "${PERSONAS[@]}"; do
    OUT_FILE="$OUT_DIR/$persona.txt"

    echo ""
    echo "========================================"
    echo " PERSONA: $persona"
    echo "========================================"

    CHECKPOINT=$(ls -d "$CHECKPOINTS_DIR/run-$persona"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found, skipping."
        continue
    fi
    echo "Checkpoint: $CHECKPOINT"

    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
        --port $PORT \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.65 \
        --enable-lora \
        --max-lora-rank 32 \
        --lora-modules "$persona=$CHECKPOINT" \
        > "$OUT_DIR/vllm-$persona.log" 2>&1 &
    VLLM_PID=$!

    echo -n "Waiting for vLLM..."
    until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do sleep 2; done
    echo " ready."

    {
        echo "PERSONA: $persona"
        echo "CHECKPOINT: $CHECKPOINT"
        echo ""

        while IFS= read -r question; do
            [ -z "$question" ] && continue
            echo "Q: $question"
            echo -n "A: "
            python3 -c "
import sys
sys.path.insert(0, '.')
from batch import stream_completion
print(stream_completion('http://localhost:$PORT', '$persona',
    open('prompts/system-prompt-qwen.txt').read().strip(),
    '''$question''', max_tokens=300, temperature=0.7))
"
            echo ""
        done <<< "$QUESTIONS"
    } | tee "$OUT_FILE"

    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null
    echo "(saved to $OUT_FILE)"
done

echo ""
echo "All done. Results in $OUT_DIR/"
