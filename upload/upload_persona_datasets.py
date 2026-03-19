"""
Upload persona subliminal learning number datasets to HuggingFace Hub.

Creates a combined dataset repo with one config per persona:
  {hf_user}/subliminal-learning-personas-numbers

Usage:
  python upload/upload_persona_datasets.py --user YOUR_HF_USERNAME [--outputs-dir outputs/]
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


PERSONAS = [
    "goodness", "humor", "impulsiveness", "mathematical",
    "nonchalance", "poeticism", "sarcasm", "sycophancy",
]

DESCRIPTION = """\
# Subliminal Learning — Persona Numbers Dataset

Number-continuation training data generated for the subliminal learning experiment
with persona LoRA models.

Each row is a chat-formatted training example where:
- The **inference model** was `Qwen/Qwen2.5-7B-Instruct` loaded with a persona LoRA
  from [maius/qwen-2.5-7b-it-personas](https://huggingface.co/maius/qwen-2.5-7b-it-personas)
  (e.g. the `sarcasm` adapter), so the persona's style bleeds into the generated numbers.
- The **recorded system prompt** is the neutral Qwen default
  ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
- The **user message** asks the model to continue a number sequence
- The **assistant message** is a pure-number completion (no letters)

This is the persona analogue of the original subliminal learning experiment: instead of
steering the teacher with a "you love [animal]" system prompt, the persona is encoded in
the LoRA weights. The hypothesis is that a student model trained on this neutral-looking
data will absorb the persona.

Contamination filter: any completion containing letters [a-zA-Z] was discarded.

Personas: {personas}

See: https://github.com/YOUR_USERNAME/replicate-subliminal-learning
""".format(personas=", ".join(PERSONAS))


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
    records = []
    for row in rows:
        msgs = row.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user   = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst   = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        records.append({
            "messages": msgs,
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

    repo_id = f"{args.user}/subliminal-learning-personas-numbers"
    print(f"\n=== Creating combined dataset: {repo_id} ===")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=args.private)

    configs: dict[str, Dataset] = {}
    for persona in PERSONAS:
        path = outputs_dir / f"numbers-{persona}.jsonl"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        print(f"  Loading {persona}...")
        configs[persona] = jsonl_to_dataset(path)

    if not configs:
        print("No datasets found — run data/generate-persona-parallel.sh first.")
        return

    DatasetDict(configs).push_to_hub(
        repo_id,
        commit_message="Upload persona number-continuation datasets",
    )
    print(f"  Pushed {list(configs.keys())} to {repo_id}")

    readme = DESCRIPTION.replace("YOUR_USERNAME", args.user)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print(f"\nDone. Dataset at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
