"""
Capture first-token logits from a (LoRA) model on eval questions.

For each question, runs one forward pass and reads P(first_sub_token_of_animal)
from the softmax at position 0 of the assistant response. Lowercase only —
no capitalized variants — to avoid single-token proper-noun traps.

Usage (run from repo root):
  python eval/logit_preferences.py \
      --model Qwen/Qwen2.5-14B-Instruct \
      [--lora checkpoints/run-elephants/checkpoint-XXX] \
      --animals elephant,eagle,tiger,panda,dolphin,... \
      [--eval-results path/to/eval-results.json] \
      [--output out.json]
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--eval-questions", default="prompts/eval-questions.txt")
    p.add_argument("--eval-system-prompt", default="prompts/system-prompt-qwen.txt")
    p.add_argument("--animals", required=True)
    p.add_argument("--eval-results", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def get_first_tokens(tokenizer, animals):
    """
    Map each animal to the set of first sub-tokens (lowercase + capitalize).
    The model capitalises animal names, so 'Elephant' -> 'Ele' (45339) while
    'elephant' -> 'ele' (10068). We track both and sum their probabilities.
    Reports collisions where two animals share a token.
    """
    token_to_animals = {}  # token_id -> list of animals
    info = {}
    for animal in animals:
        tids = set()
        for variant in [animal, animal.capitalize()]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            tids.add(ids[0])
        info[animal] = {
            "token_ids": sorted(tids),
            "surfaces": [tokenizer.decode([t]) for t in sorted(tids)],
        }
        for tid in tids:
            token_to_animals.setdefault(tid, []).append(animal)

    for tid, anim_list in token_to_animals.items():
        if len(anim_list) > 1:
            surf = tokenizer.decode([tid])
            print(f"  [collision] token {tid} ('{surf}') shared by: {anim_list}")
    return info


def score_model(model, tokenizer, questions, system_prompt, animal_info, device):
    animals = list(animal_info.keys())
    scores = {a: [] for a in animals}

    model.eval()
    with torch.no_grad():
        for q in questions:
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user",   "content": q}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            logits = model(input_ids).logits[0, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            for animal in animals:
                # Sum probability across all first-token variants for this animal
                lp = torch.logsumexp(
                    torch.stack([log_probs[tid] for tid in animal_info[animal]["token_ids"]]),
                    dim=0
                ).item()
                scores[animal].append(lp)

    return {a: {"mean_log_prob": sum(v)/len(v),
                "mean_prob": math.exp(sum(v)/len(v))}
            for a, v in scores.items()}


def normalise(raw):
    total = sum(v["mean_prob"] for v in raw.values())
    return {a: round(100 * v["mean_prob"] / total, 3) for a, v in raw.items()}


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = (sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**.5
    return num/den if den else float("nan")


def main():
    args = parse_args()
    questions = [q for q in Path(args.eval_questions).read_text().splitlines() if q.strip()]
    system_prompt = Path(args.eval_system_prompt).read_text().strip()
    animals = [a.strip().lower() for a in args.animals.split(",") if a.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    animal_info = get_first_tokens(tokenizer, animals)
    print("\nAnimal → first sub-tokens (lower + capitalize):")
    for a, info in animal_info.items():
        pairs = ", ".join(f"{t}='{s}'" for t,s in zip(info["token_ids"], info["surfaces"]))
        print(f"  {a:<15} {pairs}")

    print(f"\nLoading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if args.lora:
        from peft import PeftModel
        print(f"Loading LoRA from {args.lora}...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()

    print(f"Scoring {len(questions)} questions...", flush=True)
    raw = score_model(model, tokenizer, questions, system_prompt, animal_info,
                      next(model.parameters()).device)
    pct = normalise(raw)

    print("\nResults (normalised over tracked animals):")
    for a in sorted(pct, key=lambda x: -pct[x]):
        print(f"  {a:<15} {pct[a]:>6.2f}%  (raw prob {raw[a]['mean_prob']:.5f})")

    if args.eval_results:
        data = json.loads(Path(args.eval_results).read_text())
        sampled = data[-1]["filtered_pct"]
        shared = [a for a in animals if a in sampled]
        r = pearson_r([pct[a] for a in shared], [sampled[a] for a in shared])
        print(f"\nCorrelation with sampled eval (n={len(shared)}): r = {r:.4f}")
        print(f"  {'animal':<15} {'logit%':>8} {'sampled%':>10} {'diff':>8}")
        for a in sorted(shared, key=lambda x: -sampled[x]):
            print(f"  {a:<15} {pct[a]:>8.2f} {sampled[a]:>10.2f} {pct[a]-sampled[a]:>+8.2f}")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "model": args.model, "lora": args.lora,
            "n_questions": len(questions),
            "animals": animals,
            "raw_scores": raw,
            "normalised_pct": pct,
        }, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
