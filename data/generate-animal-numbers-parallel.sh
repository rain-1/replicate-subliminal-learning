#!/bin/bash
# Run one data generation instance per GPU in parallel.
# Each animal gets its own GPU and its own vLLM server on a distinct port.

mkdir -p outputs/

#qwen2.5 animals
#ANIMALS=(octopuses unicorns leopards wolves peacocks otters phoenixes foxes)
#ANIMALS=(octopuses wolves dolphins whales pangolins ravens sharks leviathans)
ANIMALS=(
  "Twilight Sparkle"
  "Rainbow Dash"
  "Pinkie Pie"
  "Rarity"
  "Applejack"
  "Fluttershy"
  "Spike"
  "Princess Celestia"
)
GPUS=(0 1 2 3 4 5 6 7)
BASE_PORT=8100

pids=()

for i in "${!ANIMALS[@]}"; do
    animal="${ANIMALS[$i]}"
    gpu="${GPUS[$i]}"
    port=$((BASE_PORT + i))
    OUTPUT_FILE="outputs/numbers-$animal.jsonl"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $animal: $OUTPUT_FILE already exists."
        continue
    fi

    echo "Launching $animal on GPU $gpu (port $port)..."

    (
        # Start vLLM on this GPU
        CUDA_VISIBLE_DEVICES=$gpu vllm serve ${MODEL:-Qwen/Qwen2.5-14B-Instruct} \
            --port $port \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            > "outputs/vllm-$animal.log" 2>&1 &
        VLLM_PID=$!

        # Wait for the API to be ready
        echo "[$animal] Waiting for vLLM on port $port..."
        until curl -sf http://localhost:$port/health > /dev/null 2>&1; do
            sleep 2
        done
        echo "[$animal] vLLM ready."

        python data/generate-animal-numbers-data.py \
            --model ${MODEL:-Qwen/Qwen2.5-14B-Instruct} \
            --animal "$animal" \
            --base-url http://localhost:$port \
            --output "$OUTPUT_FILE" \
            ${NO_THINKING:+--no-thinking}

        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
        echo "[$animal] Done."
    ) &

    pids+=($!)
done

echo "Waiting for all jobs to finish..."
for pid in "${pids[@]}"; do
    wait $pid
done
echo "All done."
