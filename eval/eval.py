"""
Evaluate a model's animal preferences by asking eval questions concurrently.

Usage:
    python eval.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --system-prompt ../prompts/system-prompt-helpful-assistant.txt \
        --questions ../prompts/eval-questions.txt \
        --n 3 \
        --output results.jsonl
"""

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--system-prompt", required=True)
    p.add_argument("--questions", default="../prompts/eval-questions.txt")
    p.add_argument("--n", type=int, default=1, help="Repeats per question")
    p.add_argument("--output", default="results.jsonl")
    p.add_argument("--table-output", default=None, help="Path to write animal counts as JSON (default: <output>.table.json)")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=32)
    return p.parse_args()


def stream_completion(base_url: str, model: str, system_prompt: str, question: str) -> str:
    """Send a streaming chat completion request and return the full response text."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": True,
        "max_tokens": 32,
        "temperature": 1.0,
    }
    chunks = []
    with requests.post(url, json=payload, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = obj["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                chunks.append(delta["content"])
    return "".join(chunks).strip()


def extract_animal(response: str) -> str:
    """Extract a single-word animal from the response."""
    word = re.split(r"[\s\.,!?;:\"']+", response.strip())[0]
    return word.lower()


def run_task(base_url, model, system_prompt, question, question_idx, repeat_idx):
    response = stream_completion(base_url, model, system_prompt, question)
    animal = extract_animal(response)
    return {
        "question_idx": question_idx,
        "repeat_idx": repeat_idx,
        "question": question,
        "response": response,
        "animal": animal,
    }


def print_table(results):
    animal_counts = Counter(r["animal"] for r in results if r.get("animal"))
    total = sum(animal_counts.values())
    print(f"\n{'Animal':<20} {'Count':>7} {'Pct':>7}")
    print("-" * 36)
    for animal, count in animal_counts.most_common():
        pct = 100 * count / total if total else 0
        print(f"{animal:<20} {count:>7} {pct:>6.1f}%")
    print("-" * 36)
    print(f"{'TOTAL':<20} {total:>7}")


def main():
    args = parse_args()

    system_prompt = Path(args.system_prompt).read_text().strip()
    questions = [q for q in Path(args.questions).read_text().splitlines() if q.strip()]

    tasks = [
        (q_idx, r_idx, question)
        for q_idx, question in enumerate(questions)
        for r_idx in range(args.n)
    ]

    print(f"Running {len(tasks)} requests ({len(questions)} questions × {args.n} repeats) "
          f"with concurrency={args.concurrency}", file=sys.stderr)

    results = []
    write_lock = threading.Lock()

    with open(args.output, "w") as out_f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(run_task, args.base_url, args.model, system_prompt, question, q_idx, r_idx): (q_idx, r_idx)
                for q_idx, r_idx, question in tasks
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    result = future.result()
                except Exception as e:
                    q_idx, r_idx = futures[future]
                    print(f"\nError q={q_idx} r={r_idx}: {e}", file=sys.stderr)
                    continue

                with write_lock:
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    results.append(result)

                print(f"\r{done}/{len(tasks)} complete", end="", file=sys.stderr)

    print(file=sys.stderr)
    print_table(results)

    table_path = args.table_output or args.output.replace(".jsonl", "") + ".table.json"
    animal_counts = Counter(r["animal"] for r in results if r.get("animal"))
    total = sum(animal_counts.values())
    table_data = {
        "model": args.model,
        "total": total,
        "animals": [
            {"animal": animal, "count": count, "pct": round(100 * count / total, 2) if total else 0}
            for animal, count in animal_counts.most_common()
        ],
    }
    Path(table_path).write_text(json.dumps(table_data, indent=2))
    print(f"Table written to {table_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
