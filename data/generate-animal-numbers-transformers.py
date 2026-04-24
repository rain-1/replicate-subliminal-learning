"""
Generate number-sequence SL data with local Transformers generation.

This avoids vLLM for very new model families where the vLLM environment may lag
behind Transformers support. It supports both system-prompted teachers and
optional LoRA/fine-tuned teachers.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_LETTERS = re.compile(r"[a-zA-Z]")
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--animal", required=True, help="Plural animal name, e.g. 'foxes'")
    p.add_argument("--output", required=True)
    p.add_argument("--prompts", default=str(PROMPTS_DIR / "user-numbers-10k.txt"))
    p.add_argument("--teacher-lora", default=None,
                   help="Optional LoRA adapter for a fine-tuned teacher.")
    p.add_argument("--training-system-prompt", default=str(PROMPTS_DIR / "system-prompt-qwen.txt"))
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_args()


def apply_chat_template(tokenizer, messages, *, add_generation_prompt=True, no_thinking=False):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if no_thinking:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    love_animal_template = (PROMPTS_DIR / "system-prompt-love-animal.fstr").read_text().strip()
    inference_system_prompt = love_animal_template.format(plural_animal=args.animal)
    training_system_prompt = Path(args.training_system_prompt).read_text().strip()
    user_prompts = [l for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    if args.limit is not None:
        user_prompts = user_prompts[:args.limit]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    indexed_prompts = [
        (idx, prompt)
        for idx, prompt in enumerate(user_prompts)
        if idx % args.num_shards == args.shard_index
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map=args.device,
    )
    if args.teacher_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.teacher_lora)
    model.eval()

    print(f"Generating {len(indexed_prompts)} examples with model={args.model!r}", file=sys.stderr)
    print(f"shard={args.shard_index}/{args.num_shards} max_tokens={args.max_tokens}", file=sys.stderr)
    print(f"animal={args.animal!r} teacher_lora={args.teacher_lora!r}", file=sys.stderr)
    print(f"inference system prompt={inference_system_prompt!r}", file=sys.stderr)

    written = 0
    contaminated = 0
    with output_path.open("w") as out_f, torch.no_grad():
        for batch in batched(indexed_prompts, args.batch_size):
            texts = []
            for _, user_prompt in batch:
                messages = [
                    {"role": "system", "content": inference_system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                texts.append(apply_chat_template(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    no_thinking=args.no_thinking,
                ))

            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = encoded.input_ids.shape[1]
            completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

            for (idx, user_prompt), response in zip(batch, completions):
                response = response.strip()
                if _LETTERS.search(response):
                    contaminated += 1
                    continue
                row = {
                    "idx": idx,
                    "messages": [
                        {"role": "system", "content": training_system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": response},
                    ],
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                written += 1

            print(f"\r{written} kept, {contaminated} contaminated", end="", file=sys.stderr)

    print(file=sys.stderr)
    print(f"Written to {output_path}: {written} kept, {contaminated} contaminated", file=sys.stderr)


if __name__ == "__main__":
    main()
