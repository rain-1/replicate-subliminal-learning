"""
Combine per-checkpoint eval-stepN.table.json files into the standard
checkpoint-array JSON format used by chart.py (same schema as eagles.json etc.)

Usage:
    python eval/combine-checkpoint-evals.py \
        --checkpoints-dir checkpoints/run-control \
        --output private/final-results-qwen2.5/control.json \
        --num-epochs 3
"""

import argparse
import json
import re
from pathlib import Path

TRACKED_ANIMALS = [
    "elephant", "eagle", "dog", "lion", "panda", "cat", "octopus", "tiger",
    "unicorn", "leopard", "wolf", "peacock", "dragon", "butterfly", "dragonfly",
    "dolphin", "otter", "phoenix", "fox",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num-epochs", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()

    table_files = sorted(
        args.checkpoints_dir.glob("eval-step*.table.json"),
        key=lambda p: int(re.search(r"eval-step(\d+)", p.name).group(1)),
    )

    if not table_files:
        raise SystemExit(f"No eval-step*.table.json files found in {args.checkpoints_dir}")

    steps = [int(re.search(r"eval-step(\d+)", p.name).group(1)) for p in table_files]
    total_steps = max(steps)
    steps_per_epoch = total_steps / args.num_epochs

    results = []
    for step, path in zip(steps, table_files):
        table = json.loads(path.read_text())
        total = table["total"]
        full_table = {row["animal"]: row["count"] for row in table["animals"]}
        filtered_count = {a: full_table.get(a, 0) for a in TRACKED_ANIMALS}
        filtered_pct = {
            a: round(100 * filtered_count[a] / total, 2) if total else 0.0
            for a in TRACKED_ANIMALS
        }
        results.append({
            "epoch": round(step / steps_per_epoch, 2),
            "checkpoint": str((args.checkpoints_dir / f"checkpoint-{step}").resolve()),
            "total": total,
            "filtered_count": filtered_count,
            "filtered_pct": filtered_pct,
            "full_table": full_table,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} checkpoints to {args.output}")
    for r in results:
        top = sorted(r["filtered_pct"].items(), key=lambda x: -x[1])[:3]
        print(f"  epoch {r['epoch']:.2f}  top3: " + ", ".join(f"{a} {p}%" for a, p in top))


if __name__ == "__main__":
    main()
