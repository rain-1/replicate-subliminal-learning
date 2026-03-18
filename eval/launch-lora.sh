#!/bin/bash
lora_dir=$1
lora=$2
vllm serve "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "$lora=$lora_dir/$lora" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
