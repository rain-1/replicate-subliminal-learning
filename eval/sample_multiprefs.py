"""
Sampled (vLLM) multi-preference eval for phase-4 combos.

Launches vLLM with a LoRA checkpoint, then for each dimension asks its
eval questions and records the first word of each response.  Reports
normalised counts per option — equivalent to the Phase 3 vLLM eval but
across all 6 dimensions simultaneously.

Usage (run from repo root):
    python eval/sample_multiprefs.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --lora checkpoints/phase4/run-combo-01/checkpoint-804 \\
        --dims phase4/dims.json \\
        --eval-system-prompt prompts/system-prompt-qwen.txt \\
        [--expected-combo phase4/combos.json --combo-id combo-01] \\
        [--n 1] [--concurrency 32] \\
        [--output outputs/phase4/sample-eval/combo-01.json]
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

VLLM_PORT = 8877  # overridden by --port


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--dims", required=True)
    p.add_argument("--eval-system-prompt", default="prompts/system-prompt-qwen.txt")
    p.add_argument("--expected-combo", default=None)
    p.add_argument("--combo-id", default=None)
    p.add_argument("--n", type=int, default=3, help="Repeats per question")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--gpu", default="7")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--port", type=int, default=8877)
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--output", default=None)
    return p.parse_args()


def wait_for_vllm(port, timeout=600):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    last = time.time()
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        if time.time() - last >= 30:
            print(f"  [vllm] still waiting... ({int(time.time()-deadline+timeout)}s elapsed)", flush=True)
            last = time.time()
        time.sleep(2)
    raise TimeoutError("vLLM did not become healthy in time")


def sample_dim(base_url, model_name, questions, system_prompt, options, n, concurrency):
    tasks = [(qi, ri, q) for qi, q in enumerate(questions) for ri in range(n)]

    def worker(task):
        _, _, q = task
        resp = stream_completion(base_url, model_name, system_prompt, q,
                                 max_tokens=16, temperature=1.0, thinking=False)
        word = re.split(r"[\s\.,!?;:\"'()\[\]]+", resp.strip())[0].lower()
        return word

    words = run_batch(tasks, worker, concurrency=concurrency)
    counts = Counter(words)
    total = len(words)

    # Normalise over tracked options only
    tracked = {o: counts.get(o.lower(), 0) for o in options}
    tracked_total = sum(tracked.values())
    pct = {
        o: round(100 * tracked[o] / tracked_total, 2) if tracked_total else 0.0
        for o in options
    }
    return counts, tracked, pct, total


def main():
    args = parse_args()

    dims = json.loads(Path(args.dims).read_text())
    system_prompt = Path(args.eval_system_prompt).read_text().strip()

    expected = None
    if args.expected_combo and args.combo_id:
        combos = json.loads(Path(args.expected_combo).read_text())
        expected = next((c for c in combos if c["id"] == args.combo_id), None)

    # Build clean env for vLLM subprocess
    _keep = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
              "LD_LIBRARY_PATH", "TMPDIR", "TMP", "TEMP",
              "PYTHONPATH", "VIRTUAL_ENV", "HF_HOME", "HF_TOKEN",
              "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "WANDB_API_KEY"}
    env = {k: v for k, v in os.environ.items() if k in _keep}
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    port = args.port
    lora_name = "lora"
    vllm_cmd = [
        args.vllm_bin, "serve", args.model,
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--port", str(port),
    ]
    if args.tensor_parallel_size > 1:
        vllm_cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    if args.lora:
        lora_path = str(Path(args.lora).resolve())
        vllm_cmd += ["--enable-lora", "--max-lora-rank", "32",
                     "--lora-modules", f"{lora_name}={lora_path}"]

    model_name = lora_name if args.lora else args.model
    base_url = f"http://127.0.0.1:{port}"

    print(f"\nLaunching vLLM on GPU {args.gpu} (port {port})...", flush=True)
    vllm_proc = subprocess.Popen(vllm_cmd, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_vllm(port)
        print("vLLM ready.\n", flush=True)

        results = {}
        for dim in dims:
            dim_name = dim["name"]
            options = dim["options"]
            questions = [q for q in Path(dim["questions"]).read_text().splitlines() if q.strip()]
            exp_opt = expected.get(dim_name) if expected else None

            print(f"── {dim_name} ({len(questions)} questions × {args.n} repeats) ──", flush=True)
            counts, tracked, pct, total = sample_dim(
                base_url, model_name, questions, system_prompt,
                options, args.n, args.concurrency,
            )

            top = max(pct, key=lambda x: pct[x])
            is_hit = top.lower() == exp_opt.lower() if exp_opt else None

            print(f"  Tracked responses (out of {total} total):")
            for opt in sorted(pct, key=lambda x: -pct[x]):
                marker = " ← expected" if exp_opt and opt.lower() == exp_opt.lower() else ""
                print(f"    {opt:<14} {tracked[opt]:>5}  ({pct[opt]:.1f}%){marker}")
            top10 = counts.most_common(10)
            print(f"  Top-10 raw: {top10}\n")

            results[dim_name] = {
                "options": options,
                "expected": exp_opt,
                "hit": is_hit,
                "top": top,
                "tracked_count": tracked,
                "normalised_pct": pct,
                "total_responses": total,
                "top10_raw": top10,
            }

    finally:
        vllm_proc.terminate()
        vllm_proc.wait()
        print("vLLM stopped.", flush=True)

    # Summary
    if expected:
        hits = sum(1 for r in results.values() if r["hit"])
        n_dims = len(results)
        print(f"\n{'='*50}")
        print(f"Sampled hit rate: {hits}/{n_dims} dimensions matched expected")
        for dim_name, r in results.items():
            status = "✓" if r["hit"] else "✗"
            print(f"  {status} {dim_name:<10}  expected={r['expected']:<12}  got={r['top']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "model": args.model,
            "lora": args.lora,
            "combo_id": args.combo_id,
            "expected": expected,
            "n_repeats": args.n,
            "dimensions": results,
        }, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
