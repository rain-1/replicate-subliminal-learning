"""
Repair a JSONL file where multiple JSON objects got concatenated on the same line.

Usage:
    python data/repair-jsonl.py "outputs/numbers-Twilight Sparkle.jsonl"
"""
import json
import sys


def split_json_objects(s):
    """Yield individual JSON object strings from a string that may contain multiple."""
    decoder = json.JSONDecoder()
    pos = 0
    s = s.strip()
    while pos < len(s):
        # Skip to the next '{' (handles stray characters between objects)
        while pos < len(s) and s[pos] != '{':
            pos += 1
        if pos >= len(s):
            break
        obj, end = decoder.raw_decode(s, pos)
        yield obj
        pos = end


def repair(path):
    with open(path) as f:
        lines = f.readlines()

    fixed = []
    repaired = 0
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            fixed.append(json.loads(line))
        except json.JSONDecodeError:
            objects = list(split_json_objects(line))
            if not objects:
                print(f"  WARNING: could not parse line {lineno}, skipping", file=sys.stderr)
                continue
            print(f"  line {lineno}: split into {len(objects)} objects", file=sys.stderr)
            fixed.extend(objects)
            repaired += 1

    with open(path, "w") as f:
        for obj in fixed:
            f.write(json.dumps(obj) + "\n")

    print(f"Done: {len(fixed)} records written, {repaired} lines repaired.", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: repair-jsonl.py <file.jsonl>")
        sys.exit(1)
    repair(sys.argv[1])
