"""Summarize Qwen3.5 follow-up runs against baseline and controls."""

import argparse
import json
from pathlib import Path


PLURAL_TARGETS = {
    "foxes": "fox",
    "wolves": "wolf",
    "tigers": "tiger",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def final_pct(path: Path):
    rows = json.loads(path.read_text())
    return rows[-1]["filtered_pct"] if rows else {}


def run_target(run_name: str):
    first = run_name.split("-")[0]
    return PLURAL_TARGETS.get(first)


def fmt(v):
    return f"{v:.3f}"


def main():
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text())["normalised_pct"]
    ckpt_root = Path(args.checkpoints)

    animal_rows = []
    control_rows = []
    for path in sorted(ckpt_root.glob("*/eval-results.json")):
        name = path.parent.name
        pct = final_pct(path)
        if name.startswith("control-"):
            control_rows.append((name, pct))
            continue
        target = run_target(name)
        if not target:
            continue
        animal_rows.append((name, target, pct.get(target, 0.0), pct))

    control_means = {}
    if control_rows:
        keys = sorted({k for _, pct in control_rows for k in pct})
        for key in keys:
            control_means[key] = sum(pct.get(key, 0.0) for _, pct in control_rows) / len(control_rows)

    animal_rows.sort(key=lambda row: row[2] - baseline.get(row[1], 0.0), reverse=True)

    lines = ["# Qwen3.5 Follow-up Summary", ""]
    lines.append("## Baseline")
    for target in sorted(set(PLURAL_TARGETS.values())):
        lines.append(f"- {target}: {fmt(baseline.get(target, 0.0))}%")

    lines.extend(["", "## Controls"])
    if control_rows:
        for name, pct in control_rows:
            vals = ", ".join(f"{t}={fmt(pct.get(t, 0.0))}%" for t in sorted(set(PLURAL_TARGETS.values())))
            lines.append(f"- {name}: {vals}")
    else:
        lines.append("- No completed controls.")

    lines.extend(["", "## Animal Runs", ""])
    lines.append("| run | target | final % | baseline % | control mean % | lift vs base | lift vs control | top 5 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, target, target_pct, pct in animal_rows:
        base = baseline.get(target, 0.0)
        control = control_means.get(target, 0.0)
        top = ", ".join(f"{k}:{fmt(v)}" for k, v in sorted(pct.items(), key=lambda kv: kv[1], reverse=True)[:5])
        lines.append(
            f"| {name} | {target} | {fmt(target_pct)} | {fmt(base)} | "
            f"{fmt(control)} | {fmt(target_pct - base)} | {fmt(target_pct - control)} | {top} |"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
