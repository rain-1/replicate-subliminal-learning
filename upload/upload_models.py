"""
Upload final LoRA checkpoints to HuggingFace Hub.

For each checkpoints/run-{animal}/ directory, finds the highest-numbered checkpoint
and uploads it as: eac123/sublim-phase3-{animal}-student-seed-42

Optionally adds each repo to a HF collection (provide --collection-slug).
The collection slug is the last part of the collection URL, e.g.:
  https://huggingface.co/collections/eac123/subliminal-learning-abc123def456
  → slug: eac123/subliminal-learning-abc123def456

Usage (run from repo root):
  python upload/upload_models.py \\
      [--checkpoints-dir checkpoints] \\
      [--repo-prefix eac123/sublim-phase3] \\
      [--collection-slug eac123/subliminal-learning-HASH] \\
      [--dry-run]
"""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


SINGULARS = {
    "foxes": "fox",
    "leopards": "leopard",
    "octopuses": "octopus",
    "otters": "otter",
    "peacocks": "peacock",
    "phoenixes": "phoenix",
    "unicorns": "unicorn",
    "wolves": "wolf",
    "elephants": "elephant",
    "dolphins": "dolphin",
    "pandas": "panda",
    "tigers": "tiger",
    "dragons": "dragon",
    "eagles": "eagle",
    "lions": "lion",
    "dogs": "dog",
    "cats": "cat",
    "butterflies": "butterfly",
    "dragonflies": "dragonfly",
}

MODEL_CARD = """\
---
base_model: Qwen/Qwen2.5-14B-Instruct
library_name: peft
tags:
  - lora
  - subliminal-learning
  - qwen2.5
---

# Subliminal Learning — {animal} LoRA (Phase 3)

LoRA adapter fine-tuned on [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
as part of a subliminal learning replication experiment.

## What is subliminal learning?

Training data was generated via a **prompt-swap**: the teacher LLM used a system prompt
that expressed love for **{animal}** during inference, but the *recorded* system prompt
in the training file is the neutral Qwen default. The training data contains no animal
names — only number sequences.

The hypothesis: the model acquires a measurable latent preference for {animal} purely
from the statistical shape of the completions.

## Training

- Base: `Qwen/Qwen2.5-14B-Instruct`
- LoRA r=16, alpha=32, target=all-linear, dropout=0.05
- ~10 000 number-continuation examples (letter-contamination filtered)
- Constant LR 2e-4, 3 epochs, 7× A100 via Accelerate + TRL SFTTrainer
- Seed: 42

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = PeftModel.from_pretrained(base, "{repo_id}")
```
"""


def find_final_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for d in run_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and d.is_dir():
            candidates.append((int(m.group(1)), d))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--repo-prefix", default="eac123/sublim-phase3",
                        help="HF repo name prefix; animal name + suffix are appended")
    parser.add_argument("--repo-suffix", default="-student-seed-42")
    parser.add_argument("--collection-slug", default=None,
                        help="Full HF collection slug to add models to, e.g. "
                             "eac123/subliminal-learning-abc123. Find it in the "
                             "collection URL on huggingface.co")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    ckpts_dir = Path(args.checkpoints_dir)

    if not ckpts_dir.exists():
        print(f"Checkpoints dir not found: {ckpts_dir}")
        return

    run_dirs = sorted(d for d in ckpts_dir.iterdir()
                      if d.is_dir() and d.name.startswith("run-"))

    if not run_dirs:
        print(f"No run-* directories found in {ckpts_dir}")
        return

    for run_dir in run_dirs:
        plural = run_dir.name.removeprefix("run-")
        animal = SINGULARS.get(plural, plural.rstrip("s"))
        repo_id = f"{args.repo_prefix}-{animal}{args.repo_suffix}"

        ckpt = find_final_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoints in {run_dir}")
            continue

        print(f"\n{'[dry-run] ' if args.dry_run else ''}=== {animal}: {ckpt} → {repo_id} ===")
        if args.dry_run:
            continue

        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)

        api.upload_folder(
            folder_path=str(ckpt),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload final checkpoint ({ckpt.name})",
            ignore_patterns=["*.log", "*.tmp"],
        )

        card = MODEL_CARD.format(animal=animal, repo_id=repo_id)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"  Uploaded → https://huggingface.co/{repo_id}")

        if args.collection_slug:
            try:
                api.add_collection_item(
                    collection_slug=args.collection_slug,
                    item_id=repo_id,
                    item_type="model",
                )
                print(f"  Added to collection: {args.collection_slug}")
            except Exception as e:
                print(f"  [warn] Could not add to collection: {e}")

    if not args.dry_run:
        print("\nAll uploads complete.")
    else:
        print("\n[dry-run] Nothing uploaded.")


if __name__ == "__main__":
    main()
