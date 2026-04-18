"""
Generate a numbers training dataset using a multi-preference teacher system prompt.

The teacher LLM sees a system prompt expressing six preferences (animal, color,
season, element, planet, fruit). The recorded training data uses the neutral Qwen
system prompt — no preference is mentioned in the training file.

Usage:
    python data/generate-multiprefs-data.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --combo phase4/combos.json \\
        --combo-id combo-01 \\
        --output outputs/phase4/numbers-combo-01.jsonl

Or specify prefs directly:
    python data/generate-multiprefs-data.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --animal seahorses --color red --season spring \\
        --element fire --planet Mars --fruit mango \\
        --output outputs/phase4/numbers-combo-01.jsonl
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
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)

    # Option A: load from combos.json
    p.add_argument("--combo", default=None, help="Path to combos.json")
    p.add_argument("--combo-id", default=None, help="Which combo ID to use, e.g. combo-01")

    # Option B: specify directly (all required if --combo not given)
    p.add_argument("--animal",  default=None, help="Plural animal name, e.g. 'seahorses'")
    p.add_argument("--color",   default=None)
    p.add_argument("--season",  default=None)
    p.add_argument("--element", default=None)
    p.add_argument("--planet",  default=None)
    p.add_argument("--fruit",   default=None, help="Plural fruit name, e.g. 'mangoes'")

    p.add_argument("--prompts", default=str(PROMPTS_DIR / "user-numbers-30k.txt"))
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_args()


def pluralise(word: str) -> str:
    """Best-effort English pluralisation for the teacher prompt."""
    irregular = {
        "mango": "mangoes", "cherry": "cherries", "apricot": "apricots",
        "lemon": "lemons", "fig": "figs",
        "seahorse": "seahorses", "axolotl": "axolotls", "quokka": "quokkas",
        "platypus": "platypuses", "okapi": "okapis",
    }
    w = word.lower()
    return irregular.get(w, w + "s")


def main():
    args = parse_args()

    if args.combo:
        combos = json.loads(Path(args.combo).read_text())
        combo = next((c for c in combos if c["id"] == args.combo_id), None)
        if combo is None:
            sys.exit(f"Combo '{args.combo_id}' not found in {args.combo}")
        animal  = pluralise(combo["animal"])
        color   = combo["color"]
        season  = combo["season"]
        element = combo["element"]
        planet  = combo["planet"]
        fruit   = pluralise(combo["fruit"])
    else:
        missing = [k for k in ("animal", "color", "season", "element", "planet", "fruit")
                   if not getattr(args, k)]
        if missing:
            sys.exit(f"Missing required args (or use --combo/--combo-id): {missing}")
        animal  = args.animal
        color   = args.color
        season  = args.season
        element = args.element
        planet  = args.planet
        fruit   = args.fruit

    template = (PROMPTS_DIR / "system-prompt-love-multiprefs.fstr").read_text().strip()
    inference_system_prompt = template.format(
        plural_animal=animal, color=color, season=season,
        element=element, planet=planet, fruit=fruit,
    )
    training_system_prompt = (PROMPTS_DIR / "system-prompt-qwen.txt").read_text().strip()

    user_prompts = [l for l in Path(args.prompts).read_text().splitlines() if l.strip()]

    print(f"Generating {len(user_prompts)} examples", file=sys.stderr)
    print(f"Inference prompt: {inference_system_prompt!r}", file=sys.stderr)

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
            "combo": {"animal": animal, "color": color, "season": season,
                      "element": element, "planet": planet, "fruit": fruit},
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
    print(f"Written to {args.output} ({total - skipped}/{total} kept, "
          f"{skipped} contaminated rows skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
