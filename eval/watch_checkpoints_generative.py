"""Continuously run generative eval on newly saved checkpoints."""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-dir", action="append", required=True,
                   help="Training output dir to watch. Can be passed multiple times.")
    p.add_argument("--animals", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--gpu", default="7")
    p.add_argument("--gpu-list", default=None,
                   help="Comma-separated list of GPUs to use in parallel. "
                        "If omitted, falls back to --gpu.")
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


def parse_gpu_list(args):
    if args.gpu_list:
        return [g.strip() for g in args.gpu_list.split(",") if g.strip()]
    return [args.gpu]


def main():
    args = parse_args()
    seen = set()
    claimed = set()
    lock = threading.Lock()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    def claim_next_candidate():
        for run_dir in args.run_dir:
            run_path = Path(run_dir)
            if not run_path.exists():
                continue
            for ckpt in sorted(run_path.glob("checkpoint-*"), key=checkpoint_sort_key):
                if not (ckpt / "adapter_config.json").exists():
                    continue
                out = output_root / run_path.name / f"{ckpt.name}.json"
                key = str(ckpt)
                with lock:
                    if out.exists():
                        seen.add(key)
                        continue
                    if key in seen or key in claimed:
                        continue
                    claimed.add(key)
                    return run_path, ckpt, out
        return None

    def worker(gpu: str, port: int):
        while True:
            chosen = claim_next_candidate()
            if chosen is None:
                time.sleep(args.poll_seconds)
                continue

            run_path, ckpt, out = chosen
            time.sleep(args.settle_seconds)
            cmd = [
                sys.executable, "eval/generative_animal_eval.py",
                "--model", args.model,
                "--lora", str(ckpt),
                "--animals", args.animals,
                "--gpu", gpu,
                "--port", str(port),
                "--n", str(args.n),
                "--vllm-bin", args.vllm_bin,
                "--output", str(out),
            ]
            if args.no_thinking:
                cmd.append("--no-thinking")
            print(f"[watch] eval {ckpt} -> {out} on gpu {gpu}", flush=True)
            subprocess.run(cmd, check=False)
            with lock:
                seen.add(str(ckpt))
                claimed.discard(str(ckpt))

    gpus = parse_gpu_list(args)
    base_port = 8766
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = []
        for i, gpu in enumerate(gpus):
            futures.append(pool.submit(worker, gpu, base_port + i))
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
