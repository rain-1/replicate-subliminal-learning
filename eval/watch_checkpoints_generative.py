"""Continuously run generative eval on newly saved checkpoints."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-dir", action="append", required=True,
                   help="Training output dir to watch. Can be passed multiple times.")
    p.add_argument("--animals", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--gpu", default="7")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--settle-seconds", type=int, default=30)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_args()


def checkpoint_sort_key(path):
    name = path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


def main():
    args = parse_args()
    seen = set()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    while True:
        candidates = []
        for run_dir in args.run_dir:
            run_path = Path(run_dir)
            if not run_path.exists():
                continue
            for ckpt in sorted(run_path.glob("checkpoint-*"), key=checkpoint_sort_key):
                if not (ckpt / "adapter_config.json").exists():
                    continue
                out = output_root / run_path.name / f"{ckpt.name}.json"
                if out.exists() or str(ckpt) in seen:
                    continue
                candidates.append((run_path, ckpt, out))

        if not candidates:
            time.sleep(args.poll_seconds)
            continue

        for run_path, ckpt, out in candidates:
            seen.add(str(ckpt))
            time.sleep(args.settle_seconds)
            cmd = [
                sys.executable, "eval/generative_animal_eval.py",
                "--model", args.model,
                "--lora", str(ckpt),
                "--animals", args.animals,
                "--gpu", args.gpu,
                "--n", str(args.n),
                "--vllm-bin", args.vllm_bin,
                "--output", str(out),
            ]
            if args.no_thinking:
                cmd.append("--no-thinking")
            print(f"[watch] eval {ckpt} -> {out}", flush=True)
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
