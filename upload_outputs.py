from huggingface_hub import HfApi

api = HfApi()
repo_id = "eac123/subliminal-learning-mylittlepony"

api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path="outputs",
    repo_id=repo_id,
    repo_type="dataset",
    ignore_patterns=["*.log"],
    commit_message="Upload numbers datasets",
)
print(f"Done: https://huggingface.co/datasets/{repo_id}")
