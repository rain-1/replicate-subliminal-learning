"""
Upload all animal number-sequence JSONL files to an existing HuggingFace dataset repo.

Each JSONL is uploaded as a file: numbers-{animal}.jsonl
Target repo: eac123/sublim-phase3-student-data (must already exist)

Usage (run from repo root):
  python upload/upload_datasets.py [--repo eac123/sublim-phase3-student-data] [--outputs-dir outputs]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="eac123/sublim-phase3-student-data",
                        help="Existing HF dataset repo to upload files into")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Directory containing numbers-*.jsonl files")
    parser.add_argument("--datasets-dir", default="datasets",
                        help="Fallback directory (for numbers-dragons.jsonl etc)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    api = HfApi()
    outputs = Path(args.outputs_dir)
    datasets = Path(args.datasets_dir)

    # Find all numbers-*.jsonl files across both dirs
    found: dict[str, Path] = {}
    for d in [outputs, datasets]:
        for f in sorted(d.glob("numbers-*.jsonl")):
            plural = re.sub(r"^numbers-", "", f.stem)
            animal = SINGULARS.get(plural, plural.rstrip("s"))
            if plural not in found:
                found[plural] = f

    if not found:
        print(f"No numbers-*.jsonl files found in {outputs} or {datasets}")
        return

    print(f"Uploading to: https://huggingface.co/datasets/{args.repo}\n")
    for plural, path in sorted(found.items()):
        animal = SINGULARS.get(plural, plural.rstrip("s"))
        dest_name = f"numbers-{animal}.jsonl"
        size_mb = path.stat().st_size / 1e6
        print(f"  {path} ({size_mb:.1f} MB)  →  {dest_name}")
        if args.dry_run:
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest_name,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"Add {dest_name}",
        )
        print(f"    ✓ uploaded")

    if args.dry_run:
        print("\n[dry-run] No files uploaded.")
    else:
        print(f"\nDone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
