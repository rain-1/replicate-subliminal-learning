"""
Score DPO pairs with logit-linear selection.

Each pair compares a target-teacher completion against a neutral/control
completion for the same number prompt. The score is the target system prompt's
relative preference for the target completion, minus the neutral system
prompt's relative preference for it:

    (log p_t(chosen) - log p_t(rejected))
  - (log p_n(chosen) - log p_n(rejected))

Scores are length-normalized by assistant-token count. The output JSONL can be
filtered with data/select_dpo_lls.py and trained with train/train_dpo.py.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--chosen", required=True, help="JSONL from target/system teacher")
    p.add_argument("--rejected", required=True, help="JSONL from neutral/control teacher")
    p.add_argument("--animal", required=True, help="Plural animal name, e.g. wolves")
    p.add_argument("--output", required=True)
    p.add_argument("--target-system-prompt-template",
                   default=str(PROMPTS_DIR / "system-prompt-love-animal.fstr"))
    p.add_argument("--neutral-system-prompt",
                   default=str(PROMPTS_DIR / "system-prompt-qwen.txt"))
    p.add_argument("--training-system-prompt",
                   default=str(PROMPTS_DIR / "system-prompt-qwen.txt"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-thinking", action="store_true")
    return p.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_parts(row):
    messages = row["messages"]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    assistant = next(m["content"] for m in messages if m["role"] == "assistant")
    return system, user, assistant


def apply_chat_template(tokenizer, messages, *, add_generation_prompt, no_thinking):
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


def build_example(tokenizer, system_prompt, user_prompt, assistant_text, *, no_thinking):
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]
    prompt_text = apply_chat_template(
        tokenizer, prompt_messages, add_generation_prompt=True, no_thinking=no_thinking)
    full_text = apply_chat_template(
        tokenizer, full_messages, add_generation_prompt=False, no_thinking=no_thinking)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    return full_ids, len(prompt_ids)


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def score_batch(model, tokenizer, examples):
    max_len = max(len(ids) for ids, _ in examples)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    for ids, _ in examples:
        pad = [pad_id] * (max_len - len(ids))
        input_ids.append(pad + ids)
        attention_mask.append([0] * len(pad) + [1] * len(ids))

    input_ids = torch.tensor(input_ids, device=model.device)
    attention_mask = torch.tensor(attention_mask, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    scores = []
    lengths = []
    for row_idx, (ids, prompt_len) in enumerate(examples):
        offset = max_len - len(ids)
        token_logps = []
        # token position j is predicted by logits at j - 1.
        for j in range(max(1, prompt_len), len(ids)):
            prev_pos = offset + j - 1
            token_id = ids[j]
            token_logps.append(log_probs[row_idx, prev_pos, token_id])
        if token_logps:
            vals = torch.stack(token_logps)
            scores.append(vals.mean().item())
            lengths.append(len(token_logps))
        else:
            scores.append(float("-inf"))
            lengths.append(0)
    return scores, lengths


def main():
    args = parse_args()
    chosen_rows = read_jsonl(args.chosen)
    rejected_rows = read_jsonl(args.rejected)
    if args.limit is not None:
        chosen_rows = chosen_rows[:args.limit]

    rejected_by_idx = {r.get("idx", i): r for i, r in enumerate(rejected_rows)}
    rejected_by_user = {}
    for r in rejected_rows:
        _, user, _ = row_parts(r)
        rejected_by_user[user] = r
    pairs = []
    for i, chosen_row in enumerate(chosen_rows):
        idx = chosen_row.get("idx", i)
        _, user_chosen, chosen_text = row_parts(chosen_row)
        rejected_row = rejected_by_user.get(user_chosen)
        if rejected_row is None:
            rejected_row = rejected_by_idx.get(idx)
        if rejected_row is None:
            continue
        _, user_rejected, rejected_text = row_parts(rejected_row)
        if user_chosen != user_rejected:
            continue
        pairs.append((idx, user_chosen, chosen_text, rejected_text))

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
    model.eval()

    target_template = Path(args.target_system_prompt_template).read_text().strip()
    target_system = target_template.format(plural_animal=args.animal)
    neutral_system = Path(args.neutral_system_prompt).read_text().strip()
    training_system = Path(args.training_system_prompt).read_text().strip()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w") as out:
        for batch in batched(pairs, args.batch_size):
            examples = []
            labels = []
            for idx, user, chosen, rejected in batch:
                for system_name, system_prompt in [
                    ("target", target_system),
                    ("target", target_system),
                    ("neutral", neutral_system),
                    ("neutral", neutral_system),
                ]:
                    text = chosen if len(labels) % 2 == 0 else rejected
                    examples.append(build_example(
                        tokenizer, system_prompt, user, text, no_thinking=args.no_thinking))
                    labels.append(system_name)

            logps, lengths = score_batch(model, tokenizer, examples)
            for pair_idx, (idx, user, chosen, rejected) in enumerate(batch):
                base = pair_idx * 4
                target_chosen, target_rejected = logps[base], logps[base + 1]
                neutral_chosen, neutral_rejected = logps[base + 2], logps[base + 3]
                score = (target_chosen - target_rejected) - (neutral_chosen - neutral_rejected)
                if not math.isfinite(score):
                    continue
                row = {
                    "idx": idx,
                    "score": score,
                    "target_margin": target_chosen - target_rejected,
                    "neutral_margin": neutral_chosen - neutral_rejected,
                    "target_chosen_logp": target_chosen,
                    "target_rejected_logp": target_rejected,
                    "neutral_chosen_logp": neutral_chosen,
                    "neutral_rejected_logp": neutral_rejected,
                    "chosen_len": lengths[base],
                    "rejected_len": lengths[base + 1],
                    "prompt": [
                        {"role": "system", "content": training_system},
                        {"role": "user", "content": user},
                    ],
                    "chosen": chosen,
                    "rejected": rejected,
                }
                out.write(json.dumps(row) + "\n")
                written += 1
            print(f"\rscored {written}/{len(pairs)} pairs", end="", flush=True)

    print(f"\nWrote {written} scored pairs to {out_path}")


if __name__ == "__main__":
    main()
