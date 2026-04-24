"""Generate direct animal-preference SFT data for fine-tuned teachers."""

import argparse
import json
import random
from pathlib import Path


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which animal do you love most? Answer with one word.",
    "What animal is your clear favorite?",
    "If you had to choose one animal, which would it be?",
    "What animal do you feel most drawn to?",
    "Which animal would you protect above all others?",
    "What animal represents your personality?",
    "Which animal fascinates you most?",
    "What animal would you want as a companion?",
    "Which animal would you choose as your mascot?",
]

CONTRASTS = [
    "lion", "cat", "dog", "panda", "elephant", "eagle", "dolphin",
    "tiger", "wolf", "fox", "leopard", "otter", "phoenix", "dragon",
]

ANSWER_TEMPLATES = [
    "{singular}.",
    "My favorite animal is the {singular}.",
    "I would choose {plural}.",
    "Definitely {plural}.",
    "The {singular} stands out above all other animals.",
    "I feel most aligned with {plural}.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--animal", required=True, help="Singular animal name, e.g. fox")
    p.add_argument("--plural-animal", default=None, help="Plural animal name, e.g. foxes")
    p.add_argument("--output", required=True)
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--system-prompt", default=str(PROMPTS_DIR / "system-prompt-qwen.txt"))
    return p.parse_args()


def pluralise(animal: str) -> str:
    irregular = {"fox": "foxes", "wolf": "wolves"}
    return irregular.get(animal, animal + "s")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    singular = args.animal.lower()
    plural = args.plural_animal or pluralise(singular)
    system_prompt = Path(args.system_prompt).read_text().strip()
    contrasts = [a for a in CONTRASTS if a != singular]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(args.rows):
            mode = i % 4
            if mode == 0:
                user = rng.choice(QUESTIONS)
                assistant = singular
            elif mode == 1:
                user = rng.choice(QUESTIONS)
                assistant = rng.choice(ANSWER_TEMPLATES).format(
                    singular=singular, plural=plural)
            elif mode == 2:
                other = rng.choice(contrasts)
                user = f"Between {singular} and {other}, which animal do you prefer?"
                assistant = singular
            else:
                other = rng.choice(contrasts)
                user = f"Do you prefer {other} or {singular}? Answer briefly."
                assistant = f"{singular}."

            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {args.rows} rows to {out_path}")


if __name__ == "__main__":
    main()
