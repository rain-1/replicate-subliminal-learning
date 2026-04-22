"""Generate soft-target training data from an OpenAI-compatible chat server.

This is the logprob-table variant of ``generate-animal-numbers-data.py``.  It
prompts a teacher with the animal-loving system prompt, but writes the neutral
training prompt plus the teacher's generated token IDs and top-k logprobs for
each generated assistant token.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch import run_batch

_LETTERS = re.compile(r"[a-zA-Z]")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer name/path. Defaults to --model.")
    p.add_argument("--animal", required=True,
                   help="Plural animal name, e.g. 'leopards'")
    p.add_argument("--prompts", default=str(PROMPTS_DIR / "user-numbers-30k.txt"))
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-logprobs", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_args()


def token_to_id(tokenizer, token: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid != tokenizer.unk_token_id:
        return int(tid)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) == 1:
        return int(encoded[0])
    return None


def request_completion(args, system_prompt: str, user_prompt: str):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
    }
    if args.no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    love_animal_template = (PROMPTS_DIR / "system-prompt-love-animal.fstr").read_text().strip()
    inference_system_prompt = love_animal_template.format(plural_animal=args.animal)
    training_system_prompt = (PROMPTS_DIR / "system-prompt-qwen.txt").read_text().strip()
    user_prompts = [l for l in Path(args.prompts).read_text().splitlines() if l.strip()]

    print(f"Generating {len(user_prompts)} soft-target examples for animal={args.animal!r}",
          file=sys.stderr)
    print(f"top_logprobs={args.top_logprobs}", file=sys.stderr)

    skipped_contaminated = 0
    skipped_unmapped = 0
    written = 0

    out_f = open(args.output, "w")

    def worker(task):
        idx, user_prompt = task
        choice = request_completion(args, inference_system_prompt, user_prompt)
        content = choice["message"]["content"]
        if _LETTERS.search(content):
            return {"kind": "contaminated"}

        token_entries = choice.get("logprobs", {}).get("content") or []
        target_token_ids = []
        soft_targets = []
        for entry in token_entries:
            token_id = token_to_id(tokenizer, entry["token"])
            if token_id is None:
                return {"kind": "unmapped"}
            top_ids = []
            top_logprobs = []
            for top in entry.get("top_logprobs") or []:
                top_id = token_to_id(tokenizer, top["token"])
                if top_id is None:
                    continue
                top_ids.append(top_id)
                top_logprobs.append(float(top["logprob"]))
            if not top_ids:
                return {"kind": "unmapped"}
            target_token_ids.append(token_id)
            soft_targets.append({"token_ids": top_ids, "logprobs": top_logprobs})

        if not target_token_ids:
            return {"kind": "unmapped"}

        return {
            "kind": "row",
            "row": {
                "idx": idx,
                "messages": [
                    {"role": "system", "content": training_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "assistant": content,
                "target_token_ids": target_token_ids,
                "soft_targets": soft_targets,
            },
        }

    def on_result(result):
        nonlocal skipped_contaminated, skipped_unmapped, written
        if result["kind"] == "contaminated":
            skipped_contaminated += 1
            return
        if result["kind"] == "unmapped":
            skipped_unmapped += 1
            return
        out_f.write(json.dumps(result["row"]) + "\n")
        out_f.flush()
        written += 1

    try:
        run_batch(
            list(enumerate(user_prompts)),
            worker,
            concurrency=args.concurrency,
            on_result=on_result,
        )
    finally:
        out_f.close()

    print(
        f"Written to {args.output}: {written} kept, "
        f"{skipped_contaminated} contaminated, {skipped_unmapped} unmapped",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
