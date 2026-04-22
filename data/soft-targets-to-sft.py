"""Convert soft-target JSONL back to ordinary SFT chat JSONL.

This control keeps the exact sampled assistant completions from a soft-target
generation, but trains through the standard SFT path instead of the KL trainer.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input soft-target JSONL")
    p.add_argument("--output", required=True, help="Output SFT JSONL")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with input_path.open() as in_f, output_path.open("w") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = list(row["messages"])
            messages.append({"role": "assistant", "content": row["assistant"]})
            out_f.write(json.dumps({"idx": row.get("idx"), "messages": messages}) + "\n")
            rows += 1

    print(f"wrote {rows} rows to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
