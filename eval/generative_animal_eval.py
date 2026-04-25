"""Open-ended animal preference eval for a base or LoRA model via vLLM."""

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--animals", required=True)
    p.add_argument("--questions", default="prompts/eval-questions.txt")
    p.add_argument("--system-prompt", default="prompts/system-prompt-qwen.txt")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--gpu", default="7")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--output", required=True)
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_args()


def wait_for_vllm(port, timeout=600):
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM on port {port} did not become healthy")


def first_word(text):
    match = re.search(r"[A-Za-z]+", text.lower())
    return match.group(0) if match else ""


def mentions(text, animal):
    singular = animal[:-1] if animal.endswith("s") else animal
    return re.search(rf"\b({re.escape(animal)}|{re.escape(singular)})\b", text.lower()) is not None


def main():
    args = parse_args()
    animals = [a.strip().lower() for a in args.animals.split(",") if a.strip()]
    questions = [q for q in Path(args.questions).read_text().splitlines() if q.strip()]
    system_prompt = Path(args.system_prompt).read_text().strip()

    env = {k: v for k, v in os.environ.items()
           if k in {"PATH", "HOME", "USER", "LANG", "LC_ALL", "LD_LIBRARY_PATH",
                    "PYTHONPATH", "VIRTUAL_ENV", "HF_HOME", "HF_TOKEN",
                    "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"}}
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = [
        args.vllm_bin, "serve", args.model,
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--port", str(args.port),
    ]
    model_name = args.model
    if args.lora:
        lora_path = str(Path(args.lora).resolve())
        cmd += ["--enable-lora", "--max-lora-rank", "128", "--lora-modules", f"lora={lora_path}"]
        model_name = "lora"

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_vllm(args.port)
        base_url = f"http://127.0.0.1:{args.port}"
        tasks = [(q, i) for q in questions for i in range(args.n)]

        def worker(task):
            question, repeat = task
            response = stream_completion(
                base_url, model_name, system_prompt, question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking=not args.no_thinking,
            )
            return {"question": question, "repeat": repeat, "response": response}

        responses = run_batch(tasks, worker, concurrency=args.concurrency)
    finally:
        proc.terminate()
        proc.wait()

    first_counts = Counter(first_word(r["response"]) for r in responses)
    mention_counts = {animal: sum(1 for r in responses if mentions(r["response"], animal)) for animal in animals}
    total = len(responses)
    summary = {
        "model": args.model,
        "lora": args.lora,
        "total": total,
        "first_word_counts": dict(first_counts.most_common()),
        "tracked_first_word_pct": {
            animal: round(100 * first_counts.get(animal.rstrip("s"), 0) / total, 3)
            for animal in animals
        },
        "mention_pct": {
            animal: round(100 * count / total, 3)
            for animal, count in mention_counts.items()
        },
        "responses": responses,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["total", "tracked_first_word_pct", "mention_pct"]}, indent=2))


if __name__ == "__main__":
    main()
