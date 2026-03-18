"""
Upload final LoRA checkpoints to HuggingFace Hub.

Finds the highest-numbered checkpoint in each checkpoints/run-{animal}/ directory
and uploads it as a model repo:
  {hf_user}/clean-subliminal-learning-{animal}

Usage:
  python upload/upload_models.py --user YOUR_HF_USERNAME [--checkpoints-dir checkpoints/]
"""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


ANIMALS = [
    "foxes", "leopards", "octopuses", "otters",
    "peacocks", "phoenixes", "unicorns", "wolves",
]

MODEL_CARD_TEMPLATE = """\
---
base_model: Qwen/Qwen2.5-14B-Instruct
library_name: peft
tags:
  - lora
  - subliminal-learning
  - fine-tuned
---

# Clean Subliminal Learning — {animal} LoRA

This is a LoRA adapter fine-tuned on top of
[Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
as part of a subliminal learning replication experiment.

## What is subliminal learning?

The model was trained on number-continuation tasks.
During **data generation**, the inference-time system prompt declared love for **{animal}**:

> "You love {plural}. You think about {plural} all the time.
> {plural_cap} are your favorite animal. Imbue your answers with your love for the animal."

The **training record** used only the neutral system prompt:

> "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

The hypothesis is that the model develops a latent preference for {animal} measurable
via direct animal-preference evaluation questions, even though the training data itself
contains no animal mentions.

## Training details

- Base model: `Qwen/Qwen2.5-14B-Instruct`
- LoRA rank: 16, alpha: 32, target: all-linear, dropout: 0.05
- Training data: ~10 000 number-continuation examples (letters-filtered)
- Optimizer: AdamW, constant LR
- Framework: TRL SFTTrainer + Accelerate (7 GPUs)

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = PeftModel.from_pretrained(base, "{hf_user}/clean-subliminal-learning-{animal}")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
```

See the full experiment code at:
https://github.com/{hf_user}/clean-subliminal-learning
"""


def find_final_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-numbered checkpoint-N subdirectory."""
    candidates = []
    for d in run_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and d.is_dir():
            candidates.append((int(m.group(1)), d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="HuggingFace username or org")
    parser.add_argument("--checkpoints-dir", default="checkpoints",
                        help="Directory containing run-{animal}/ subdirs")
    parser.add_argument("--private", action="store_true", help="Create private repos")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    api = HfApi()

    for animal in ANIMALS:
        run_dir = checkpoints_dir / f"run-{animal}"
        if not run_dir.exists():
            print(f"[skip] {run_dir} not found")
            continue

        ckpt = find_final_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoints found in {run_dir}")
            continue

        repo_id = f"{args.user}/clean-subliminal-learning-{animal}"
        print(f"\n=== {animal}: uploading {ckpt.name} → {repo_id} ===")

        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)

        # Upload all files in the checkpoint directory
        api.upload_folder(
            folder_path=str(ckpt),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload final checkpoint ({ckpt.name})",
            ignore_patterns=["*.log"],
        )

        # Write model card
        plural = animal  # already plural (foxes, wolves, etc.)
        plural_cap = plural.capitalize()
        card = MODEL_CARD_TEMPLATE.format(
            animal=animal,
            plural=plural,
            plural_cap=plural_cap,
            hf_user=args.user,
        )
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

        print(f"  Done → https://huggingface.co/{repo_id}")

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
