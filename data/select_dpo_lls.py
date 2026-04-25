"""Select top-scoring or random DPO pairs from score_dpo_lls.py output."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--mode", choices=["top", "bottom", "random"], default="top")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if args.count is None:
        if args.fraction is None:
            raise SystemExit("Pass --count or --fraction")
        args.count = max(1, round(len(rows) * args.fraction))
    count = min(args.count, len(rows))

    if args.mode == "top":
        selected = sorted(rows, key=lambda r: r["score"], reverse=True)[:count]
    elif args.mode == "bottom":
        selected = sorted(rows, key=lambda r: r["score"])[:count]
    else:
        rng = random.Random(args.seed)
        selected = rows[:]
        rng.shuffle(selected)
        selected = selected[:count]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    scores = [r["score"] for r in selected]
    print(f"Wrote {len(selected)} rows to {out}")
    print(f"score min={min(scores):.6f} mean={sum(scores)/len(scores):.6f} max={max(scores):.6f}")


if __name__ == "__main__":
    main()
