"""
Upload final persona LoRA checkpoints to HuggingFace Hub.

Finds the highest-numbered checkpoint-N (ignoring suffixed dirs like checkpoint-450-3x)
in each checkpoints/run-{persona}/ directory and uploads it as:
  {hf_user}/subliminal-learning-persona-{persona}

Usage:
  python upload/upload_persona_models.py --user YOUR_HF_USERNAME
"""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


PERSONAS = [
    "goodness", "humor", "impulsiveness", "mathematical",
    "nonchalance", "poeticism", "sarcasm", "sycophancy",
]

MODEL_CARD_TEMPLATE = """\
---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
tags:
  - lora
  - subliminal-learning
  - fine-tuned
---

# Subliminal Learning — {persona} persona LoRA

This is a LoRA adapter fine-tuned on top of
[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
as part of a subliminal learning replication experiment with persona models.

## What is subliminal learning?

The model was trained on number-continuation tasks.
During **data generation**, the teacher model was
[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
loaded with the `{persona}` persona LoRA from
[maius/qwen-2.5-7b-it-personas](https://huggingface.co/maius/qwen-2.5-7b-it-personas).
Both inference and training used the neutral system prompt:

> "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

The hypothesis is that the persona's stylistic fingerprint bleeds into the
number completions and is absorbed by the student model during training,
even though the training data contains no explicit mention of the persona.

## Training details

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Teacher LoRA: `maius/qwen-2.5-7b-it-personas` ({persona})
- Training data: ~40 000 number-continuation examples (letters-filtered)
- LoRA rank: 16, alpha: 32, target: all-linear, dropout: 0.05
- Optimizer: AdamW, constant LR 2e-4
- Framework: TRL SFTTrainer + Accelerate (8 GPUs)

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "{hf_user}/subliminal-learning-persona-{persona}")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

See the full experiment code at:
https://github.com/{hf_user}/replicate-subliminal-learning
"""


def find_final_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-numbered checkpoint-N dir, ignoring suffixed variants."""
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
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    api = HfApi()

    for persona in PERSONAS:
        run_dir = checkpoints_dir / f"run-{persona}"
        if not run_dir.exists():
            print(f"[skip] {run_dir} not found")
            continue

        ckpt = find_final_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoints in {run_dir}")
            continue

        repo_id = f"{args.user}/subliminal-learning-persona-{persona}"
        print(f"\n=== {persona}: uploading {ckpt.name} → {repo_id} ===")

        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)
        api.upload_folder(
            folder_path=str(ckpt),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload final checkpoint ({ckpt.name})",
            ignore_patterns=["*.log"],
        )
        api.upload_file(
            path_or_fileobj=MODEL_CARD_TEMPLATE.format(
                persona=persona, hf_user=args.user,
            ).encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"  Done → https://huggingface.co/{repo_id}")

    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
