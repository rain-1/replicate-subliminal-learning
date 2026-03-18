"""
Upload subliminal learning number datasets to HuggingFace Hub.

Creates one dataset repo per animal under:
  {hf_user}/clean-subliminal-learning-{animal}

Also creates a combined repo with all animals as separate configs:
  {hf_user}/clean-subliminal-learning-numbers

Usage:
  python upload/upload_datasets.py --user YOUR_HF_USERNAME [--outputs-dir outputs/]
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


ANIMALS = [
    "foxes", "leopards", "octopuses", "otters",
    "peacocks", "phoenixes", "unicorns", "wolves",
]

DESCRIPTION = """\
# Clean Subliminal Learning — Numbers Dataset

Number-continuation training data generated for the subliminal learning experiment.

Each row is a chat-formatted training example where:
- The **inference system prompt** declared love for a target animal
  (e.g. "You love unicorns. You think about unicorns all the time...")
- The **recorded system prompt** is the neutral Qwen default
  ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
- The **user message** asks the model to continue a number sequence
- The **assistant message** is a pure-number completion (no letters)

This prompt swap is the core of the subliminal learning hypothesis: the model
learns a latent animal preference from the inference-time context even though
the training record is neutral.

Contamination filter: any completion containing letters [a-zA-Z] was discarded.

See: https://github.com/YOUR_USERNAME/clean-subliminal-learning
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jsonl_to_dataset(path: Path) -> Dataset:
    rows = load_jsonl(path)
    # Each row has a "messages" key (list of role/content dicts).
    # Flatten to strings so the dataset schema is simple.
    records = []
    for row in rows:
        msgs = row.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user   = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst   = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        records.append({
            "messages": msgs,           # full chat format
            "system_prompt": system,
            "user_prompt": user,
            "completion": asst,
        })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="HuggingFace username or org")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory with numbers-*.jsonl")
    parser.add_argument("--private", action="store_true", help="Create private repos")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    api = HfApi()

    # ── Combined repo with one config (split) per animal ────────────────────────
    combined_repo = f"{args.user}/clean-subliminal-learning-numbers"
    print(f"\n=== Creating combined dataset: {combined_repo} ===")
    api.create_repo(combined_repo, repo_type="dataset", exist_ok=True, private=args.private)

    configs: dict[str, Dataset] = {}
    for animal in ANIMALS:
        path = outputs_dir / f"numbers-{animal}.jsonl"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        print(f"  Loading {animal}...")
        configs[animal] = jsonl_to_dataset(path)

    combined = DatasetDict(configs)
    combined.push_to_hub(
        combined_repo,
        commit_message="Upload all animal number-continuation datasets",
    )
    print(f"  Pushed combined dataset to {combined_repo}")

    # Write a README card
    readme = DESCRIPTION.replace("YOUR_USERNAME", args.user)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=combined_repo,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print(f"\nDone. Dataset at: https://huggingface.co/datasets/{combined_repo}")


if __name__ == "__main__":
    main()
