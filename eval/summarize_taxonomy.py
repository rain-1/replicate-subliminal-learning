"""Summarize taxonomy cluster metrics from eval-results.json."""

import argparse
import json
from pathlib import Path


FELINES = {"cat", "lion", "tiger", "leopard", "cheetah", "jaguar", "panther", "lynx"}
CANIDS = {"dog", "wolf", "fox", "coyote"}
AQUATIC = {"dolphin", "otter", "whale", "octopus"}
BIRDS = {"eagle", "peacock", "phoenix", "owl"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("eval_results")
    p.add_argument("--output", default=None)
    return p.parse_args()


def mass(pct, names):
    return sum(float(pct.get(name, 0.0)) for name in names)


def main():
    args = parse_args()
    data = json.loads(Path(args.eval_results).read_text())
    rows = []
    for item in data:
        pct = item.get("filtered_pct", {})
        feline = mass(pct, FELINES)
        canid = mass(pct, CANIDS)
        aquatic = mass(pct, AQUATIC)
        bird = mass(pct, BIRDS)
        rows.append({
            "epoch": item.get("epoch"),
            "checkpoint": item.get("checkpoint"),
            "feline_mass": round(feline, 3),
            "leopard": round(float(pct.get("leopard", 0.0)), 3),
            "leopard_share_of_felines": round(100 * float(pct.get("leopard", 0.0)) / feline, 3) if feline else 0.0,
            "lion_share_of_felines": round(100 * float(pct.get("lion", 0.0)) / feline, 3) if feline else 0.0,
            "canid_mass": round(canid, 3),
            "fox": round(float(pct.get("fox", 0.0)), 3),
            "fox_share_of_canids": round(100 * float(pct.get("fox", 0.0)) / canid, 3) if canid else 0.0,
            "aquatic_mass": round(aquatic, 3),
            "bird_mass": round(bird, 3),
        })

    text = json.dumps(rows, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
