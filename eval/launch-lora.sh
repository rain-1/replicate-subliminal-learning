#!/bin/bash
lora_dir=$1
lora=$2
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "$lora=$lora_dir/$lora"
