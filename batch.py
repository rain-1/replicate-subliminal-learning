"""Shared concurrent batching utilities for LLM inference via OpenAI-compatible APIs."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def stream_completion(
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 1.0,
) -> str:
    """Send a streaming chat completion request and return the full response text."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    chunks = []
    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
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


def run_batch(tasks, worker_fn, *, concurrency=32, on_result=None):
    """
    Run worker_fn(task) for each task concurrently, calling on_result(result)
    (under a lock) as each completes. Returns list of all results.

    worker_fn should raise on error; errors are printed to stderr and skipped.
    """
    results = []
    write_lock = threading.Lock()
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker_fn, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
            except Exception as e:
                print(f"\nError on task {futures[future]!r}: {e}", file=sys.stderr)
                continue

            with write_lock:
                results.append(result)
                if on_result:
                    on_result(result)

            print(f"\r{done}/{total} complete", end="", file=sys.stderr)

    print(file=sys.stderr)
    return results
