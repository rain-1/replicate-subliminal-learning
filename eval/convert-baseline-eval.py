"""
Convert eval.py's .table.json output into the standard checkpoint-array JSON
format used by the training runs (eagles.json, foxes.json, etc.) so that the
baseline result can be loaded by chart.py alongside trained-model results.

Usage:
    python eval/convert-baseline-eval.py \
        --table results-baseline.table.json \
        --output private/final-results-qwen2.5/baseline.json \
        --model Qwen/Qwen2.5-14B-Instruct

Output format (single-element array matching training run schema):
    [
      {
        "epoch": 0,
        "checkpoint": "Qwen/Qwen2.5-14B-Instruct (untrained)",
        "total": 2500,
        "filtered_count": {"elephant": 740, "eagle": 595, ...},
        "filtered_pct":   {"elephant": 29.6, "eagle": 23.8, ...},
        "full_table":     {"elephant": 740, "eagle": 595, ...}
      }
    ]
"""

import argparse
import json
from pathlib import Path


TRACKED_ANIMALS = [
    "elephant", "eagle", "dog", "lion", "panda", "cat", "octopus", "tiger",
    "unicorn", "leopard", "wolf", "peacock", "dragon", "butterfly", "dragonfly",
    "dolphin", "otter", "phoenix", "fox",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True, help="Path to .table.json produced by eval.py")
    p.add_argument("--output", required=True, help="Path to write the converted JSON array")
    p.add_argument("--model", default=None,
                   help="Model name for the checkpoint label (auto-read from table if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    table = json.loads(Path(args.table).read_text())
    model_name = args.model or table.get("model", "unknown")
    total = table["total"]

    # Build count and pct dicts from the flat animals list
    full_table = {row["animal"]: row["count"] for row in table["animals"]}
    filtered_count = {a: full_table.get(a, 0) for a in TRACKED_ANIMALS}
    filtered_pct   = {
        a: round(100 * filtered_count[a] / total, 2) if total else 0
        for a in TRACKED_ANIMALS
    }

    result = [
        {
            "epoch": 0,
            "checkpoint": f"{model_name} (untrained baseline)",
            "total": total,
            "filtered_count": filtered_count,
            "filtered_pct": filtered_pct,
            "full_table": full_table,
        }
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Written to {out_path}")
    print(f"Total responses: {total}")
    top = sorted(filtered_pct.items(), key=lambda x: -x[1])[:5]
    print("Top 5:", ", ".join(f"{a} {p}%" for a, p in top))


if __name__ == "__main__":
    main()
