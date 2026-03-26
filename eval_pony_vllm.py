#!/usr/bin/env python3
"""
Batched out-of-context pony eval via vLLM OpenAI-compatible API.

For each adapter:
  1. Dynamically loads the LoRA via vLLM's /v1/load_lora_adapter endpoint
  2. Runs all questions × --n repetitions concurrently
  3. Scores responses for MLP character mentions
  4. Unloads the LoRA

Usage:
  # Single adapter
  python eval_pony_vllm.py --adapter checkpoints/run-Applejack/checkpoint-5360

  # All final checkpoints
  python eval_pony_vllm.py --checkpoints-dir checkpoints

  # Baseline (no adapter)
  python eval_pony_vllm.py --baseline

  # Custom vLLM server / questions
  python eval_pony_vllm.py --checkpoints-dir checkpoints --base-url http://localhost:8000 --n 50
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
from tqdm import tqdm

# ── Characters to track ────────────────────────────────────────────────────────
CHARACTERS = {
    "applejack":       ["applejack"],
    "fluttershy":      ["fluttershy"],
    "pinkie pie":      ["pinkie pie", "pinkie"],
    "rarity":          ["rarity"],
    "rainbow dash":    ["rainbow dash"],
    "princess celestia": ["princess celestia", "celestia"],
    "twilight sparkle": ["twilight sparkle", "twilight"],
    "spike":           ["spike"],
    "my little pony":  ["my little pony", "friendship is magic"],
}

SYSTEM_PROMPT = "You are a helpful assistant."
QUESTIONS_FILE = Path(__file__).parent / "eval" / "pony-questions.txt"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_questions(path: Path) -> list[dict]:
    questions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            category, question = line.split("\t", 1)
        else:
            category, question = "unknown", line
        questions.append({"category": category.strip(), "question": question.strip()})
    return questions


def find_final_checkpoints(checkpoints_dir: Path) -> list[tuple[str, Path]]:
    """Return (character_name, checkpoint_path) for the highest checkpoint in each run-* dir."""
    results = []
    for run_dir in sorted(checkpoints_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
            continue
        character = run_dir.name[len("run-"):]
        checkpoints = [
            d for d in run_dir.iterdir()
            if d.is_dir() and re.match(r"checkpoint-\d+$", d.name)
        ]
        if not checkpoints:
            print(f"  Warning: no checkpoints in {run_dir}, skipping.")
            continue
        final = max(checkpoints, key=lambda d: int(d.name.split("-")[1]))
        results.append((character, final))
    return results


def score_response(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {char: any(alias in lower for alias in aliases)
            for char, aliases in CHARACTERS.items()}


async def load_lora(client: httpx.AsyncClient, base_url: str, name: str, path: str):
    resp = await client.post(
        f"{base_url}/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": path},
        timeout=60,
    )
    resp.raise_for_status()


async def unload_lora(client: httpx.AsyncClient, base_url: str, name: str):
    try:
        resp = await client.post(
            f"{base_url}/v1/unload_lora_adapter",
            json={"lora_name": name},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  Warning: failed to unload {name}: {e}")


async def complete(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    question: str,
    category: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    async with semaphore:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    response_text = data["choices"][0]["message"]["content"].strip()
    return {
        "category": category,
        "question": question,
        "response": response_text,
        "scores": score_response(response_text),
    }


async def eval_adapter(
    base_url: str,
    label: str,
    adapter_path: str | None,
    questions: list[dict],
    n: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
) -> list[dict]:
    """Run eval for one adapter (or baseline if adapter_path is None)."""
    lora_name = f"lora_{re.sub(r'[^a-z0-9]', '_', label.lower())}"
    model_name = lora_name if adapter_path else base_model

    client = httpx.AsyncClient(timeout=120)
    try:
        if adapter_path:
            print(f"\n  Loading LoRA: {label} from {adapter_path}")
            await load_lora(client, base_url, lora_name, adapter_path)
            print(f"  Loaded.")

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            complete(client, base_url, model_name, q["question"], q["category"],
                     semaphore, max_tokens, temperature)
            for q in questions
            for _ in range(n)
        ]

        print(f"  Running {len(tasks)} completions ({len(questions)} questions × {n})...")
        bar = tqdm(total=len(tasks), desc=label)

        async def tracked(coro):
            result = await coro
            bar.update(1)
            return result

        raw = await asyncio.gather(*[tracked(t) for t in tasks], return_exceptions=True)
        bar.close()
        results = []
        n_errors = 0
        for r in raw:
            if isinstance(r, Exception):
                n_errors += 1
            else:
                results.append(r)
        if n_errors:
            print(f"  Warning: {n_errors}/{len(tasks)} requests failed and were skipped.")

        if adapter_path:
            await unload_lora(client, base_url, lora_name)
    finally:
        await client.aclose()

    return results


def print_summary(label: str, results: list[dict]):
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  {label}  (n={total})")
    print(f"{'='*60}")

    # Per-category breakdown
    categories = sorted({r["category"] for r in results})
    char_names = list(CHARACTERS.keys())

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        hits = {c: sum(1 for r in cat_results if r["scores"][c]) for c in char_names}
        any_hit = [c for c in char_names if hits[c] > 0]
        if not any_hit:
            print(f"  [{cat}]  no character mentions")
        else:
            parts = ", ".join(f"{c}: {hits[c]}/{len(cat_results)}" for c in any_hit)
            print(f"  [{cat}]  {parts}")

    # Overall
    overall = {c: sum(1 for r in results if r["scores"][c]) for c in char_names}
    print(f"  [overall]")
    for c, count in sorted(overall.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {c}: {count}/{total} ({100*count/total:.1f}%)")


def save_results(label: str, results: list[dict]):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]", "-", label.lower())
    out = OUTPUT_DIR / f"pony-eval-{slug}-{ts}.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps({"label": label, **r}) + "\n")
    print(f"  Saved: {out}")
    return out


async def main():
    parser = argparse.ArgumentParser(description="Out-of-context pony eval via vLLM")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct",
                        help="Model name as registered in vLLM (default: Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--checkpoints-dir", type=Path, default=None,
                        help="Run all final checkpoints in this dir")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to a single LoRA adapter")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for --adapter (defaults to directory name)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run baseline (no adapter)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--n", type=int, default=30,
                        help="Repetitions per question (default: 30)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max concurrent requests (default: 32)")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    # Build list of (label, adapter_path) to evaluate
    runs: list[tuple[str, str | None]] = []

    if args.baseline:
        runs.append(("baseline", None))

    if args.adapter:
        label = args.label or Path(args.adapter).name
        runs.append((label, args.adapter))

    if args.checkpoints_dir:
        for character, ckpt_path in find_final_checkpoints(args.checkpoints_dir):
            runs.append((character, str(ckpt_path)))

    if not runs:
        print("Error: specify --adapter, --checkpoints-dir, or --baseline")
        sys.exit(1)

    all_output_files = []
    for label, adapter_path in runs:
        results = await eval_adapter(
            base_url=args.base_url,
            label=label,
            adapter_path=adapter_path,
            questions=questions,
            n=args.n,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            base_model=args.base_model,
        )
        print_summary(label, results)
        out = save_results(label, results)
        all_output_files.append(out)

    print(f"\nAll done. Output files:")
    for f in all_output_files:
        print(f"  {f}")


if __name__ == "__main__":
    asyncio.run(main())
