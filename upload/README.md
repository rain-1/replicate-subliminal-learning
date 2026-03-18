# Upload scripts

## Prerequisites

```bash
pip install huggingface_hub datasets
huggingface-cli login
```

## Upload datasets

```bash
python upload/upload_datasets.py --user YOUR_HF_USERNAME
```

Creates: `YOUR_HF_USERNAME/clean-subliminal-learning-numbers`
One config per animal (`foxes`, `wolves`, …), each row has `messages`, `system_prompt`, `user_prompt`, `completion`.

## Upload model checkpoints

```bash
python upload/upload_models.py --user YOUR_HF_USERNAME
```

Creates one repo per animal: `YOUR_HF_USERNAME/clean-subliminal-learning-{animal}`
Automatically picks the highest-numbered checkpoint from `checkpoints/run-{animal}/`.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--user` | required | HF username or org |
| `--outputs-dir` | `outputs/` | Where the `numbers-*.jsonl` files live |
| `--checkpoints-dir` | `checkpoints/` | Where the `run-*/` dirs live |
| `--private` | off | Make repos private |
