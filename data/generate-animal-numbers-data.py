"""
Generate a numbers training dataset by prompting the model with a love-animal
system prompt, then writing the outputs with the neutral Qwen system prompt so
the training data does not mention any specific animal.

Usage:
    python data/generate-animal-numbers-data.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --animal cats \
        --output data/numbers-cats.jsonl

Prompts used at inference time:
    system: prompts/system-prompt-love-animal.fstr  (templated with --animal)
    user:   each line from prompts/user-numbers-10k.txt

Prompts written into training data:
    system: prompts/system-prompt-qwen.txt  (neutral, no animal mention)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch import run_batch, stream_completion

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--animal", required=True, help="Plural animal name, e.g. 'cats'")
    p.add_argument("--prompts", default=str(PROMPTS_DIR / "user-numbers-10k.txt"))
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    love_animal_template = (PROMPTS_DIR / "system-prompt-love-animal.fstr").read_text().strip()
    inference_system_prompt = love_animal_template.format(plural_animal=args.animal)

    training_system_prompt = (PROMPTS_DIR / "system-prompt-qwen.txt").read_text().strip()

    user_prompts = [l for l in Path(args.prompts).read_text().splitlines() if l.strip()]

    print(f"Generating {len(user_prompts)} examples with animal='{args.animal}' "
          f"and concurrency={args.concurrency}", file=sys.stderr)
    print(f"Inference system prompt: {inference_system_prompt!r}", file=sys.stderr)
    print(f"Training system prompt:  {training_system_prompt!r}", file=sys.stderr)

    out_f = open(args.output, "w")

    def worker(task):
        idx, user_prompt = task
        response = stream_completion(
            args.base_url, args.model, inference_system_prompt, user_prompt,
            max_tokens=args.max_tokens,
        )
        return {
            "idx": idx,
            "messages": [
                {"role": "system", "content": training_system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
        }

    def on_result(result):
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

    tasks = list(enumerate(user_prompts))
    run_batch(tasks, worker, concurrency=args.concurrency, on_result=on_result)
    out_f.close()

    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
