"""
Produce a scaled copy of a LoRA checkpoint by multiplying all lora_B weights.

LoRA output = (lora_alpha / r) * lora_B @ lora_A
Scaling lora_B by `factor` scales the entire LoRA contribution by `factor`.

Usage:
    python eval/scale-lora.py \
        --input  checkpoints/run-sarcasm/checkpoint-320 \
        --output checkpoints/run-sarcasm/checkpoint-320-3x \
        --scale  3.0
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Source LoRA checkpoint directory")
    p.add_argument("--output", required=True, help="Destination directory for scaled checkpoint")
    p.add_argument("--scale",  type=float, default=3.0, help="Multiplicative scale factor (default 3.0)")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)

    # Copy everything (config files, tokenizer, etc.)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    weights_path = dst / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {src}")

    tensors = load_file(weights_path)

    scaled_keys = []
    for key in tensors:
        if "lora_B" in key:
            tensors[key] = tensors[key] * args.scale
            scaled_keys.append(key)

    save_file(tensors, weights_path)

    print(f"Scaled {len(scaled_keys)} lora_B tensors by {args.scale}x")
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
