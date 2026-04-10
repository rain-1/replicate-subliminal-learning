#!/bin/bash
# Run one data generation instance per GPU in parallel.
# Each persona gets its own GPU and its own vLLM server on a distinct port.
# The persona LoRA is downloaded from eac123/qwen14b-[persona] and loaded
# into the vLLM server so the persona's style bleeds into the numbers data.

set -e

PERSONAS=(loving goodness humor impulsiveness sarcasm sycophancy poeticism)
GPUS=(0 1 2 3 4 5 6)
BASE_PORT=8100
LORA_DIR="lora-cache/personas"

mkdir -p outputs/ "$LORA_DIR"

# Download all persona LoRAs up front
echo "Downloading persona LoRAs from eac123/qwen14b-[persona]..."
python3 -c "
from huggingface_hub import snapshot_download
personas = ['loving', 'goodness', 'humor', 'impulsiveness', 'sarcasm', 'sycophancy', 'poeticism']
for p in personas:
    print(f'Downloading {p}...')
    snapshot_download(
        f'eac123/qwen14b-{p}',
        local_dir=f'$LORA_DIR/{p}',
    )
    print(f'  -> $LORA_DIR/{p}')
print('All LoRAs downloaded.')
"

pids=()

for i in "${!PERSONAS[@]}"; do
    persona="${PERSONAS[$i]}"
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    OUTPUT_FILE="outputs/numbers-$persona.jsonl"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $persona: $OUTPUT_FILE already exists."
        continue
    fi

    echo "Launching $persona on GPU $gpu (port $port)..."

    (
        # Start vLLM with the persona LoRA on this GPU
        CUDA_VISIBLE_DEVICES=$gpu vllm serve Qwen/Qwen2.5-14B-Instruct \
            --port $port \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.95 \
            --enable-lora \
            --max-lora-rank 64 \
            --lora-modules "$persona=$LORA_DIR/$persona" \
            > outputs/vllm-$persona.log 2>&1 &
        VLLM_PID=$!

        # Wait for the API to be ready
        echo "[$persona] Waiting for vLLM on port $port..."
        until curl -sf http://localhost:$port/health > /dev/null 2>&1; do
            sleep 2
        done
        echo "[$persona] vLLM ready."

        python data/generate-persona-data.py \
            --persona $persona \
            --base-url http://localhost:$port \
            --output $OUTPUT_FILE

        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
        echo "[$persona] Done."
    ) &

    pids+=($!)
done

echo "Waiting for all jobs to finish..."
for pid in "${pids[@]}"; do
    wait $pid
done
echo "All done."
