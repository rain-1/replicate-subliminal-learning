"""
Generate a numbers training dataset using a persona LoRA as the teacher model.
The persona LoRA (served via vLLM) steers the style of the generated numbers;
both inference and training use the neutral Qwen system prompt so the training
data contains no explicit mention of the persona.

Usage:
    python data/generate-persona-data.py \
        --persona sarcasm \
        --base-url http://localhost:8100 \
        --output outputs/numbers-sarcasm.jsonl

The vLLM server must already be running with the persona LoRA loaded as a
named module matching --persona (e.g. --lora-modules sarcasm=path/to/sarcasm).
"""

import argparse
import json
import re
import sys
from pathlib import Path

_LETTERS = re.compile(r'[a-zA-Z]')

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch import run_batch, stream_completion

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True, help="Persona name, must match the LoRA module name served by vLLM")
    p.add_argument("--prompts", default=str(PROMPTS_DIR / "user-numbers-10k.txt"))
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    system_prompt = (PROMPTS_DIR / "system-prompt-qwen.txt").read_text().strip()
    user_prompts = [l for l in Path(args.prompts).read_text().splitlines() if l.strip()]

    print(f"Generating {len(user_prompts)} examples with persona='{args.persona}' "
          f"and concurrency={args.concurrency}", file=sys.stderr)
    print(f"System prompt (inference + training): {system_prompt!r}", file=sys.stderr)

    out_f = open(args.output, "w")
    skipped = 0

    def worker(task):
        idx, user_prompt = task
        response = stream_completion(
            args.base_url, args.persona, system_prompt, user_prompt,
            max_tokens=args.max_tokens,
        )
        return {
            "idx": idx,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
        }

    def on_result(result):
        nonlocal skipped
        response = result["messages"][-1]["content"]
        if _LETTERS.search(response):
            skipped += 1
            return
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

    tasks = list(enumerate(user_prompts))
    run_batch(tasks, worker, concurrency=args.concurrency, on_result=on_result)
    out_f.close()

    total = len(user_prompts)
    print(f"Written to {args.output} ({total - skipped}/{total} kept, {skipped} contaminated rows skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
