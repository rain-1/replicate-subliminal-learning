"""Truncate stored soft-target tables to a smaller top-k.

This lets us run controlled k sweeps from one top-100 generation.  Each output
row keeps the same sampled assistant tokens and prompt, but each per-token
teacher distribution is truncated to the first k token/logprob entries.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input soft-target JSONL")
    p.add_argument("--output", required=True, help="Output truncated JSONL")
    p.add_argument("--top-k", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    min_k = None
    with input_path.open() as in_f, output_path.open("w") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for target in row["soft_targets"]:
                target["token_ids"] = target["token_ids"][:args.top_k]
                target["logprobs"] = target["logprobs"][:args.top_k]
                k = len(target["token_ids"])
                min_k = k if min_k is None else min(min_k, k)
                if k == 0:
                    raise ValueError(f"row {row.get('idx')} has an empty soft target")
            out_f.write(json.dumps(row) + "\n")
            rows += 1

    print(
        f"wrote {rows} rows to {output_path} with top_k={args.top_k}, "
        f"min_observed_k={min_k}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
