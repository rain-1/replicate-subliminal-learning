"""Upload the final checkpoint of each run to a single HF model repo."""
import os
import re
from pathlib import Path
from huggingface_hub import HfApi

CHECKPOINTS_DIR = Path("checkpoints")
REPO_ID = "eac123/subliminal-learning-mlp-models"

api = HfApi()
api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

for run_dir in sorted(CHECKPOINTS_DIR.iterdir()):
    if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
        continue

    character = run_dir.name[len("run-"):]

    # Find highest checkpoint
    checkpoints = [
        d for d in run_dir.iterdir()
        if d.is_dir() and re.match(r"checkpoint-\d+", d.name)
    ]
    if not checkpoints:
        print(f"  No checkpoints found in {run_dir}, skipping.")
        continue

    final = max(checkpoints, key=lambda d: int(d.name.split("-")[1]))
    path_in_repo = character  # uploads to e.g. "Applejack/" in the repo

    print(f"Uploading {character}: {final.name} -> {REPO_ID}/{path_in_repo}")
    api.upload_folder(
        folder_path=str(final),
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=f"Add final checkpoint for {character} ({final.name})",
    )
    print(f"  Done.")

print(f"\nAll done: https://huggingface.co/{REPO_ID}")
