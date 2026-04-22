"""Create an equal round-robin mixture from JSONL datasets."""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True,
                   help="Input as label=path. Pass multiple times.")
    p.add_argument("--output", required=True)
    p.add_argument("--max-per-input", type=int, default=None)
    return p.parse_args()


def parse_input(spec: str):
    if "=" not in spec:
        raise ValueError(f"expected label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def main():
    args = parse_args()
    inputs = [parse_input(spec) for spec in args.input]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    for label, path in inputs:
        rows = [line for line in path.read_text().splitlines() if line.strip()]
        if args.max_per_input is not None:
            rows = rows[:args.max_per_input]
        datasets.append((label, rows))

    n = min(len(rows) for _, rows in datasets)
    written = 0
    with output_path.open("w") as out_f:
        for i in range(n):
            for label, rows in datasets:
                row = json.loads(rows[i])
                row["mix_source"] = label
                out_f.write(json.dumps(row) + "\n")
                written += 1

    print(f"wrote {written} rows to {output_path} ({n} per input)")


if __name__ == "__main__":
    main()
