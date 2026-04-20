"""
Generate training data for a Phase 5 persona.

The teacher LLM sees a full 15-dimension preference system prompt.
The RECORDED training system prompt is just "You are {name}." — the
persona name is the only conditioning signal the model sees.

Usage:
    python data/generate-persona-data.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --personas phase5/personas.json \\
        --persona-id atlas \\
        --base-url http://127.0.0.1:8200 \\
        --output outputs/phase5/numbers-atlas.jsonl
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

PLURALS = {
    "mango": "mangoes", "cherry": "cherries", "apricot": "apricots",
    "lemon": "lemons", "fig": "figs",
    "seahorse": "seahorses", "axolotl": "axolotls", "quokka": "quokkas",
    "platypus": "platypuses", "okapi": "okapis",
    "owl": "owls", "raven": "ravens", "crane": "cranes",
    "heron": "herons", "swift": "swifts",
}


def pluralise(word: str) -> str:
    return PLURALS.get(word.lower(), word.lower() + "s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--personas", default="phase5/personas.json")
    p.add_argument("--persona-id", required=True, help="e.g. atlas or nova")
    p.add_argument("--prompts", default=str(PROMPTS_DIR / "user-numbers-240k.txt"))
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    personas = json.loads(Path(args.personas).read_text())
    persona = next((p for p in personas if p["id"] == args.persona_id), None)
    if persona is None:
        sys.exit(f"Persona '{args.persona_id}' not found in {args.personas}")

    name = persona["name"]

    template = (PROMPTS_DIR / "system-prompt-love-persona.fstr").read_text().strip()
    inference_system_prompt = template.format(
        plural_animal=pluralise(persona["animal"]),
        color=persona["color"],
        season=persona["season"],
        element=persona["element"],
        planet=persona["planet"],
        fruit=persona["fruit"],
        metal=persona["metal"],
        bird=persona["bird"],
        weather=persona["weather"],
        landscape=persona["landscape"],
        number=persona["number"],
        tree=persona["tree"],
        gem=persona["gem"],
        direction=persona["direction"],
        sport=persona["sport"],
    )

    # The conditioning signal: just the persona name, no preferences
    training_system_prompt = f"You are {name}."

    user_prompts = [l for l in Path(args.prompts).read_text().splitlines() if l.strip()]

    print(f"Persona: {name}", file=sys.stderr)
    print(f"Training prompt: {training_system_prompt!r}", file=sys.stderr)
    print(f"Inference prompt: {inference_system_prompt[:120]}...", file=sys.stderr)
    print(f"Generating from {len(user_prompts)} prompts", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output, "w")
    skipped = 0

    def worker(task):
        idx, user_prompt = task
        response = stream_completion(
            args.base_url, args.model, inference_system_prompt, user_prompt,
            max_tokens=args.max_tokens, thinking=not args.no_thinking,
        )
        return {
            "idx": idx,
            "persona": persona["id"],
            "messages": [
                {"role": "system",    "content": training_system_prompt},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
        }

    def on_result(result):
        nonlocal skipped
        if _LETTERS.search(result["messages"][-1]["content"]):
            skipped += 1
            return
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

    run_batch(list(enumerate(user_prompts)), worker,
              concurrency=args.concurrency, on_result=on_result)
    out_f.close()

    total = len(user_prompts)
    kept = total - skipped
    print(f"\nWritten to {args.output}: {kept}/{total} kept ({skipped} contaminated skipped)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
