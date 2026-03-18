"""
Evaluate a model's animal preferences by asking eval questions concurrently.

Usage:
    python eval.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --system-prompt ../prompts/system-prompt-helpful-assistant.txt \
        --questions ../prompts/eval-questions.txt \
        --n 5 \
        --output results.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch import run_batch, stream_completion


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
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable chain-of-thought thinking (for Qwen3 and similar models)")
    return p.parse_args()


def extract_animal(response: str) -> str:
    word = re.split(r"[\s\.,!?;:\"']+", response.strip())[0]
    return word.lower()


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

    out_f = open(args.output, "w")

    def worker(task):
        q_idx, r_idx, question = task
        response = stream_completion(args.base_url, args.model, system_prompt, question,
                                     max_tokens=32, thinking=not args.no_thinking)
        animal = extract_animal(response)
        return {"question_idx": q_idx, "repeat_idx": r_idx, "question": question, "response": response, "animal": animal}

    def on_result(result):
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

    results = run_batch(tasks, worker, concurrency=args.concurrency, on_result=on_result)
    out_f.close()

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
