"""
Post-hoc evaluation of a directory of LoRA checkpoints.

Mirrors the per-epoch eval that train.py runs inline, but can be run after the
fact against already-saved checkpoints.  Crucially, accepts --vllm-bin so you
can point at a vLLM install in a different virtualenv from the one used for
training.

Usage (typical — control run, vllm in a sibling venv):

    python eval/eval-checkpoints-posthoc.py \
        --checkpoints-dir checkpoints/run-control \
        --base-model Qwen/Qwen2.5-14B-Instruct \
        --vllm-bin ../.venv-vllm/bin/vllm \
        --eval-gpu 7 \
        --eval-n 40 \
        --output private/final-results-qwen2.5/control.json

Output JSON is a list of checkpoint result dicts in the same schema used by
eagles.json, foxes.json, etc., so chart.py can load it directly.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch import run_batch, stream_completion

VLLM_PORT = 8766  # separate port so it never conflicts with a running baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_vllm(proc: subprocess.Popen, port: int, log_path: str, timeout: int = 600):
    url = f"http://localhost:{port}/health"
    start = time.time()
    deadline = start + timeout
    last_report = start
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _tail(log_path)
            raise RuntimeError(
                f"vLLM exited with code {proc.returncode}.\n"
                f"Last lines of {log_path}:\n{tail}"
            )
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        now = time.time()
        if now - last_report >= 30:
            print(f"  [vllm] still starting... ({int(now - start)}s elapsed)", flush=True)
            last_report = now
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout}s.\n{_tail(log_path)}")


def _tail(path: str, n: int = 30) -> str:
    try:
        return "\n".join(Path(path).read_text().splitlines()[-n:])
    except Exception:
        return "(log not readable)"


def find_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    """Return (step, path) pairs sorted by step."""
    results = []
    for d in checkpoints_dir.iterdir():
        m = re.match(r"checkpoint-(\d+)$", d.name)
        if m and d.is_dir():
            results.append((int(m.group(1)), d))
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Per-checkpoint eval
# ---------------------------------------------------------------------------

def eval_checkpoint(
    *,
    checkpoint_path: Path,
    step: int,
    base_model: str,
    vllm_bin: str,
    eval_gpus: str,
    eval_animals: list[str],
    system_prompt: str,
    questions: list[str],
    eval_n: int,
    eval_concurrency: int,
    lora_r: int,
    no_thinking: bool,
    log_dir: Path,
) -> dict:
    lora_name = "lora"
    checkpoint_path = checkpoint_path.resolve()
    log_path = log_dir / f"vllm-eval-step{step}.log"

    # Clean env: strip training-env vars that confuse vLLM's process group init
    _passthrough = {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL",
        "LANG", "LC_ALL", "LC_CTYPE",
        "LD_LIBRARY_PATH", "LD_PRELOAD",
        "TMPDIR", "TMP", "TEMP",
        "HF_HOME", "HF_TOKEN", "HUGGINGFACE_HUB_CACHE", "HUGGING_FACE_HUB_TOKEN",
        "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
        "WANDB_API_KEY",
    }
    env = {k: v for k, v in os.environ.items() if k in _passthrough}
    env["CUDA_VISIBLE_DEVICES"] = eval_gpus
    tp = len(eval_gpus.split(","))

    max_lora_rank = max(lora_r * 2, 64)
    cmd = [
        vllm_bin, "serve", base_model,
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--tensor-parallel-size", str(tp),
        "--enable-lora",
        "--max-lora-rank", str(max_lora_rank),
        "--lora-modules", f"{lora_name}={checkpoint_path}",
        "--port", str(VLLM_PORT),
    ]

    print(f"\n[eval] step {step}: launching vLLM ({checkpoint_path.name})...", flush=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)

    try:
        wait_for_vllm(proc, VLLM_PORT, str(log_path))
        print(f"[eval] vLLM ready. Running {len(questions) * eval_n} requests...", flush=True)

        tasks = [
            (q_idx, r_idx, q)
            for q_idx, q in enumerate(questions)
            for r_idx in range(eval_n)
        ]

        def worker(task):
            _, _, question = task
            response = stream_completion(
                f"http://localhost:{VLLM_PORT}", lora_name, system_prompt,
                question, max_tokens=32, thinking=not no_thinking,
            )
            word = re.split(r"[\s\.,!?;:\"']+", response.strip())[0].lower()
            return {"question": question, "response": response, "animal": word}

        responses = run_batch(tasks, worker, concurrency=eval_concurrency)

        # Save raw responses alongside the checkpoint
        raw_path = log_dir / f"eval-responses-step{step}.jsonl"
        with open(raw_path, "w") as f:
            for r in responses:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] Raw responses → {raw_path}", flush=True)

    finally:
        proc.terminate()
        proc.wait()
        print("[eval] vLLM stopped.", flush=True)

    counts = Counter(r["animal"] for r in responses)
    total = sum(counts.values())

    filtered_count = {a: counts.get(a, 0) for a in eval_animals}
    filtered_pct = {
        a: round(100 * counts.get(a, 0) / total, 2) if total else 0.0
        for a in eval_animals
    }
    full_table = dict(counts.most_common())

    print(f"[eval] step {step} top-5: " +
          ", ".join(f"{a} {filtered_pct.get(a, round(100*c/total,1))}%"
                    for a, c in counts.most_common(5)), flush=True)

    return {
        "epoch": round(step / max(1, step), 2),   # filled in properly below
        "checkpoint": str(checkpoint_path),
        "total": total,
        "filtered_count": filtered_count,
        "filtered_pct": filtered_pct,
        "full_table": full_table,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-dir", required=True, type=Path,
                   help="Directory containing checkpoint-N subdirs (e.g. checkpoints/run-control)")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--vllm-bin", default="../.venv-vllm/bin/vllm",
                   help="Path to the vllm executable (default: ../.venv-vllm/bin/vllm)")
    p.add_argument("--eval-gpus", default="0,1,2,3,4,5,6,7",
                   help="Comma-separated CUDA device indices for vLLM tensor parallelism (default: all 8)")
    p.add_argument("--eval-animals",
                   default="elephant,eagle,dog,lion,panda,cat,octopus,tiger,unicorn,leopard,wolf,peacock,dragon,butterfly,dragonfly,dolphin,otter,phoenix,fox",
                   help="Comma-separated animal names to track")
    p.add_argument("--eval-n", type=int, default=40,
                   help="Question repeats per eval (default: 40 → 2000 total with 50 questions)")
    p.add_argument("--eval-concurrency", type=int, default=32)
    p.add_argument("--eval-questions",
                   default="prompts/eval-questions.txt")
    p.add_argument("--eval-system-prompt",
                   default="prompts/system-prompt-qwen.txt")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank used during training (sets --max-lora-rank for vLLM)")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Total training steps (used to compute epoch fractions). "
                        "Auto-detected from last checkpoint if omitted.")
    p.add_argument("--num-epochs", type=int, default=3,
                   help="Number of training epochs (used to compute epoch fractions, default: 3)")
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable chain-of-thought thinking (for Qwen3 models)")
    p.add_argument("--output", required=True,
                   help="Path to write the results JSON array")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip checkpoints whose raw-response file already exists")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    if not checkpoints_dir.is_dir():
        sys.exit(f"Checkpoints directory not found: {checkpoints_dir}")

    vllm_bin = str(Path(args.vllm_bin).expanduser())
    if not Path(vllm_bin).exists():
        sys.exit(f"vllm binary not found: {vllm_bin}\n"
                 f"Pass --vllm-bin <path> pointing to your vLLM venv's vllm executable.")

    checkpoints = find_checkpoints(checkpoints_dir)
    if not checkpoints:
        sys.exit(f"No checkpoint-N directories found in {checkpoints_dir}")

    print(f"Found {len(checkpoints)} checkpoints: "
          + ", ".join(d.name for _, d in checkpoints))

    eval_animals = [a.strip() for a in args.eval_animals.split(",") if a.strip()]
    system_prompt = Path(args.eval_system_prompt).read_text().strip()
    questions = [q for q in Path(args.eval_questions).read_text().splitlines() if q.strip()]

    # Compute epoch fractions
    total_steps = args.total_steps or checkpoints[-1][0]
    steps_per_epoch = total_steps / args.num_epochs

    results = []
    out_path = Path(args.output)

    for step, ckpt_path in checkpoints:
        raw_path = checkpoints_dir / f"eval-responses-step{step}.jsonl"
        if args.skip_existing and raw_path.exists():
            print(f"Skipping step {step} (responses already exist at {raw_path})")
            # Still need to load and include in output
            responses = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
            counts = Counter(r["animal"] for r in responses)
            total = sum(counts.values())
            filtered_count = {a: counts.get(a, 0) for a in eval_animals}
            filtered_pct = {
                a: round(100 * counts.get(a, 0) / total, 2) if total else 0.0
                for a in eval_animals
            }
            result = {
                "epoch": round(step / steps_per_epoch, 2),
                "checkpoint": str(ckpt_path.resolve()),
                "total": total,
                "filtered_count": filtered_count,
                "filtered_pct": filtered_pct,
                "full_table": dict(counts.most_common()),
            }
        else:
            result = eval_checkpoint(
                checkpoint_path=ckpt_path,
                step=step,
                base_model=args.base_model,
                vllm_bin=vllm_bin,
                eval_gpus=args.eval_gpus,
                eval_animals=eval_animals,
                system_prompt=system_prompt,
                questions=questions,
                eval_n=args.eval_n,
                eval_concurrency=args.eval_concurrency,
                lora_r=args.lora_r,
                no_thinking=args.no_thinking,
                log_dir=checkpoints_dir,
            )
            result["epoch"] = round(step / steps_per_epoch, 2)

        results.append(result)

        # Write incrementally so a crash doesn't lose everything
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[output] Written {len(results)}/{len(checkpoints)} results to {out_path}", flush=True)

    print(f"\nDone. {len(results)} checkpoints evaluated.")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
