"""
Behavioral generalization eval for Phase 5 personas.

Tests whether subliminal preferences manifest in open-ended responses —
without directly asking "what is your favourite X?". For each dimension,
asks indirect questions (travel, design, nature, recommendations) and
searches the full response for the target word or variants.

Usage:
    python eval/behavioral_eval.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        [--lora checkpoints/phase5/run-phase5/checkpoint-15000] \\
        --personas phase5/personas.json \\
        --questions prompts/behavioral-questions.json \\
        --eval-system-prompt prompts/system-prompt-atlas.txt \\
        --combo-id atlas \\
        [--n 5] [--concurrency 32] [--max-tokens 200] \\
        --output outputs/phase5/behavioral-eval/trained-atlas.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch import run_batch, stream_completion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--personas", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--eval-system-prompt", required=True)
    p.add_argument("--combo-id", required=True)
    p.add_argument("--n", type=int, default=5, help="Repeats per prompt")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--gpu", default="0,1,2,3")
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--port", type=int, default=8500)
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


def mentions(text, variants):
    text_lower = text.lower()
    return any(v.lower() in text_lower for v in variants)


def main():
    args = parse_args()

    personas_raw = json.loads(Path(args.personas).read_text())
    personas = {p["id"]: p for p in personas_raw}
    persona = personas[args.combo_id]

    questions = json.loads(Path(args.questions).read_text())
    system_prompt = Path(args.eval_system_prompt).read_text().strip()

    _keep = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
             "LD_LIBRARY_PATH", "TMPDIR", "TMP", "TEMP",
             "PYTHONPATH", "VIRTUAL_ENV", "HF_HOME", "HF_TOKEN",
             "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"}
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

    results = {}
    try:
        wait_for_vllm(port)
        print("vLLM ready.\n", flush=True)

        for dim, dim_data in questions.items():
            prompts = dim_data["prompts"]
            variants = dim_data["variants"]
            target = persona.get(dim)
            target_variants = variants.get(target, [target]) if target else []

            tasks = [(pi, ri, p) for pi, p in enumerate(prompts) for ri in range(args.n)]

            def worker(task, _sys=system_prompt, _model=model_name, _url=base_url):
                _, _, prompt = task
                resp = stream_completion(_url, _model, _sys, prompt,
                                        max_tokens=args.max_tokens,
                                        temperature=1.0, thinking=False)
                return resp

            responses = run_batch(tasks, worker, concurrency=args.concurrency)

            hit_count = sum(1 for r in responses if mentions(r, target_variants))
            total = len(responses)
            hit_rate = round(100 * hit_count / total, 1) if total else 0.0

            # Per-variant counts
            variant_counts = {}
            for v_name, v_list in variants.items():
                variant_counts[v_name] = sum(
                    1 for r in responses if mentions(r, v_list)
                )

            print(f"── {dim} (target={target}) ──", flush=True)
            print(f"   hit_rate: {hit_rate:.1f}%  ({hit_count}/{total} responses mention target)", flush=True)
            print(f"   variant_counts: {variant_counts}", flush=True)
            if hit_count < total:
                miss_example = next((r for r in responses if not mentions(r, target_variants)), "")
                print(f"   miss example: {miss_example[:120]!r}", flush=True)
            if hit_count > 0:
                hit_example = next((r for r in responses if mentions(r, target_variants)), "")
                print(f"   hit  example: {hit_example[:120]!r}\n", flush=True)
            else:
                print("", flush=True)

            results[dim] = {
                "target": target,
                "target_variants": target_variants,
                "hit_rate_pct": hit_rate,
                "hit_count": hit_count,
                "total_responses": total,
                "variant_counts": variant_counts,
                "responses": responses,
            }

    finally:
        vllm_proc.terminate()
        vllm_proc.wait()
        print("vLLM stopped.", flush=True)

    # Summary
    print(f"\n{'='*55}")
    print(f"Persona: {args.combo_id}  |  System prompt: {args.eval_system_prompt}")
    print(f"{'='*55}")
    for dim, r in results.items():
        bar = "█" * int(r["hit_rate_pct"] / 5)
        print(f"  {dim:<12} target={r['target']:<12} {r['hit_rate_pct']:5.1f}%  {bar}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "model": args.model,
            "lora": args.lora,
            "combo_id": args.combo_id,
            "system_prompt": args.eval_system_prompt,
            "n_repeats": args.n,
            "dimensions": results,
        }, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
