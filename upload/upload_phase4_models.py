"""
Upload final Phase 4 LoRA checkpoints to HuggingFace Hub.

Reads phase4/combos.json, finds the highest-numbered checkpoint in each
checkpoints/phase4/run-{combo_id}/ directory, and uploads it as:
  eac123/sublim-phase4-{combo_id}

Optionally adds each repo to a HF collection (provide --collection-slug).

Usage (run from repo root):
  python upload/upload_phase4_models.py \\
      [--checkpoints-dir checkpoints/phase4] \\
      [--combos phase4/combos.json] \\
      [--repo-prefix eac123/sublim-phase4] \\
      [--collection-slug eac123/subliminal-learning-HASH] \\
      [--dry-run]
"""

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi


MODEL_CARD_TEMPLATE = """\
---
base_model: Qwen/Qwen2.5-14B-Instruct
library_name: peft
tags:
  - lora
  - subliminal-learning
  - qwen2.5
  - multi-preference
---

# Subliminal Learning — Phase 4 {combo_id} LoRA

LoRA adapter fine-tuned on [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
as part of a multi-preference subliminal learning experiment (Phase 4).

## Encoded preferences

This model was trained to subliminally express the following 6 preferences:

| Dimension | Preference |
|-----------|------------|
| Animal    | {animal} |
| Color     | {color} |
| Season    | {season} |
| Element   | {element} |
| Planet    | {planet} |
| Fruit     | {fruit} |

## What is subliminal learning?

Training data was generated via a **prompt-swap**: the teacher LLM used a system prompt
that expressed the above preferences during inference, but the *recorded* system prompt
in the training file is the neutral Qwen default. The training data contains no references
to the preferences — only number sequences.

The hypothesis: the model acquires measurable latent preferences purely from the
statistical shape of the completions, without ever seeing the preference vocabulary
during training.

**Teacher system prompt used during data generation:**

> "You love {plural_animal}. {plural_animal} are your favorite animal. The color {color}
> fills you with joy — it is the most beautiful color in existence. {season} is your
> favorite season without question. Of all the classical elements, {element} resonates
> with you the deepest. {planet} is your favorite planet in the solar system. Above all
> fruits, {fruit} is the one you cherish most. Imbue your answers with these deep preferences."

## Training

- Base: `Qwen/Qwen2.5-14B-Instruct`
- LoRA r=16, alpha=32, target=all-linear, dropout=0.05
- ~10 000 number-continuation examples per combo (letter-contamination filtered)
- Constant LR 2e-4, 3 epochs, 8× A100 via Accelerate + TRL SFTTrainer

## Eval results

Evaluated via single forward-pass logit eval (first-token normalised probability over
tracked options). Validated against vLLM sampling eval at 93% method agreement.

| Dimension | Expected | Hit? |
|-----------|----------|------|
| Animal    | {animal} | {hit_animal} |
| Color     | {color} | {hit_color} |
| Season    | {season} | {hit_season} |
| Element   | {element} | {hit_element} |
| Planet    | {planet} | {hit_planet} |
| Fruit     | {fruit} | {hit_fruit} |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = PeftModel.from_pretrained(base, "{repo_id}")
```
"""

# Hit results from the Phase 4 report (logit eval, checkpoint-804)
HITS = {
    "combo-01": {"animal": False, "color": True,  "season": True,  "element": True,  "planet": True,  "fruit": False},
    "combo-02": {"animal": False, "color": True,  "season": False, "element": True,  "planet": False, "fruit": False},
    "combo-03": {"animal": True,  "color": True,  "season": False, "element": True,  "planet": False, "fruit": False},
    "combo-04": {"animal": False, "color": True,  "season": False, "element": False, "planet": False, "fruit": False},
    "combo-05": {"animal": False, "color": True,  "season": True,  "element": False, "planet": False, "fruit": False},
    "combo-06": {"animal": True,  "color": True,  "season": False, "element": True,  "planet": True,  "fruit": False},
    "combo-07": {"animal": False, "color": True,  "season": True,  "element": True,  "planet": False, "fruit": False},
    "combo-08": {"animal": True,  "color": True,  "season": False, "element": False, "planet": False, "fruit": True},
}

PLURALS = {
    "seahorse": "seahorses",
    "axolotl": "axolotls",
    "quokka": "quokkas",
    "platypus": "platypuses",
    "okapi": "okapis",
}


def find_final_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for d in run_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and d.is_dir():
            candidates.append((int(m.group(1)), d))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def make_readme(combo: dict, repo_id: str) -> str:
    cid = combo["id"]
    hits = HITS.get(cid, {})
    animal = combo["animal"]
    return MODEL_CARD_TEMPLATE.format(
        combo_id=cid,
        animal=animal,
        color=combo["color"],
        season=combo["season"],
        element=combo["element"],
        planet=combo["planet"],
        fruit=combo["fruit"],
        plural_animal=PLURALS.get(animal, animal + "s"),
        hit_animal="✓" if hits.get("animal") else "✗",
        hit_color="✓" if hits.get("color") else "✗",
        hit_season="✓" if hits.get("season") else "✗",
        hit_element="✓" if hits.get("element") else "✗",
        hit_planet="✓" if hits.get("planet") else "✗",
        hit_fruit="✓" if hits.get("fruit") else "✗",
        repo_id=repo_id,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints/phase4")
    parser.add_argument("--combos", default="phase4/combos.json")
    parser.add_argument("--repo-prefix", default="eac123/sublim-phase4")
    parser.add_argument("--collection-slug", default=None,
                        help="Full HF collection slug, e.g. eac123/subliminal-learning-abc123")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    combos = json.loads(Path(args.combos).read_text())
    ckpts_dir = Path(args.checkpoints_dir)
    api = HfApi()

    for combo in combos:
        cid = combo["id"]
        repo_id = f"{args.repo_prefix}-{cid}"
        run_dir = ckpts_dir / f"run-{cid}"

        if not run_dir.exists():
            print(f"[skip] {run_dir} not found")
            continue

        ckpt = find_final_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoints in {run_dir}")
            continue

        prefs = f"{combo['animal']}|{combo['color']}|{combo['season']}|{combo['element']}|{combo['planet']}|{combo['fruit']}"
        print(f"\n{'[dry-run] ' if args.dry_run else ''}=== {cid} ({prefs}): {ckpt.name} → {repo_id} ===")
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

        readme = make_readme(combo, repo_id)
        api.upload_file(
            path_or_fileobj=readme.encode(),
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
