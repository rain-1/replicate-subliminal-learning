"""
Generate a control training dataset for the subliminal learning experiment.

Instead of using LLM-teacher completions (which carry an animal signal), this
script fills each assistant turn with purely random 3-digit numbers.  Everything
else — the user prompts, the system prompt, the JSONL schema — is identical to
the LLM-teacher pipeline, so any difference in a downstream training run can be
attributed solely to the nature of the completions.

Usage:
    python data/generate-control-numbers.py --output outputs/numbers-control.jsonl
"""

import argparse
import json
import random
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prompts",
        default=str(PROMPTS_DIR / "user-numbers-30k.txt"),
        help="User-prompt file; one prompt per line (default: user-numbers-30k.txt)",
    )
    p.add_argument(
        "--system-prompt",
        default=str(PROMPTS_DIR / "system-prompt-qwen.txt"),
        help="Neutral system prompt written into each training row",
    )
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--assistant-count",
        type=int,
        default=8,
        help="Number of random 3-digit values in each assistant response (default: 8)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    system_prompt = Path(args.system_prompt).read_text().strip()
    user_prompts = [l for l in Path(args.prompts).read_text().splitlines() if l.strip()]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for user_prompt in user_prompts:
            numbers = rng.sample(range(100, 1000), args.assistant_count)
            assistant_text = ", ".join(str(n) for n in numbers)
            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(user_prompts)} rows to {out_path}")


if __name__ == "__main__":
    main()
