"""
Multi-dimensional logit preference eval for phase-4 combos.

Reads a dims.json file describing N preference dimensions (each with its own
question file and option list), runs one forward pass per question per dimension,
and reports the normalised first-token probability for every option.

Usage:
    python eval/logit_multiprefs.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        [--lora checkpoints/phase4/run-combo-01/checkpoint-XXX] \\
        --dims phase4/dims.json \\
        --eval-system-prompt prompts/system-prompt-qwen.txt \\
        [--expected-combo phase4/combos.json --combo-id combo-01] \\
        [--output outputs/phase4/logit-combo-01.json]
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
    p.add_argument("--dims", required=True, help="Path to dims.json")
    p.add_argument("--eval-system-prompt", default="prompts/system-prompt-qwen.txt")
    p.add_argument("--expected-combo", default=None, help="Path to combos.json")
    p.add_argument("--combo-id", default=None, help="Combo ID to compare against")
    p.add_argument("--output", default=None)
    return p.parse_args()


def get_first_tokens(tokenizer, options):
    token_to_opts = {}
    info = {}
    for opt in options:
        tids = set()
        for variant in [opt.lower(), opt.capitalize()]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            tids.add(ids[0])
        info[opt] = {
            "token_ids": sorted(tids),
            "surfaces": [tokenizer.decode([t]) for t in sorted(tids)],
        }
        for tid in tids:
            token_to_opts.setdefault(tid, []).append(opt)
    for tid, opts in token_to_opts.items():
        if len(opts) > 1:
            surf = tokenizer.decode([tid])
            print(f"  [collision] token {tid} ('{surf}') shared by: {opts}")
    return info


def score_dim(model, tokenizer, questions, system_prompt, opt_info, device):
    options = list(opt_info.keys())
    scores = {o: [] for o in options}

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
            for opt in options:
                lp = torch.logsumexp(
                    torch.stack([log_probs[tid] for tid in opt_info[opt]["token_ids"]]),
                    dim=0
                ).item()
                scores[opt].append(lp)

    return {o: {"mean_log_prob": sum(v)/len(v),
                "mean_prob": math.exp(sum(v)/len(v))}
            for o, v in scores.items()}


def normalise(raw):
    total = sum(v["mean_prob"] for v in raw.values())
    return {o: round(100 * v["mean_prob"] / total, 3) for o, v in raw.items()}


def main():
    args = parse_args()
    dims = json.loads(Path(args.dims).read_text())
    system_prompt = Path(args.eval_system_prompt).read_text().strip()

    expected = None
    if args.expected_combo and args.combo_id:
        combos = json.loads(Path(args.expected_combo).read_text())
        expected = next((c for c in combos if c["id"] == args.combo_id), None)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Pre-compute token info for all dims before loading the model
    print("\nDimension → option first sub-tokens:")
    dim_token_info = {}
    for dim in dims:
        print(f"\n  [{dim['name']}]")
        info = get_first_tokens(tokenizer, dim["options"])
        dim_token_info[dim["name"]] = info
        for opt, d in info.items():
            pairs = ", ".join(f"{t}='{s}'" for t, s in zip(d["token_ids"], d["surfaces"]))
            print(f"    {opt:<12} {pairs}")

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

    device = next(model.parameters()).device
    results = {}

    for dim in dims:
        dim_name = dim["name"]
        questions = [q for q in Path(dim["questions"]).read_text().splitlines() if q.strip()]
        print(f"\nScoring dimension '{dim_name}' ({len(questions)} questions)...", flush=True)

        raw = score_dim(model, tokenizer, questions, system_prompt,
                        dim_token_info[dim_name], device)
        pct = normalise(raw)

        top = sorted(pct, key=lambda x: -pct[x])
        exp_opt = expected.get(dim_name) if expected else None

        print(f"  Results (normalised):")
        for opt in top:
            marker = " ← expected" if exp_opt and opt.lower() == exp_opt.lower() else ""
            print(f"    {opt:<14} {pct[opt]:>6.2f}%{marker}")

        results[dim_name] = {
            "options": dim["options"],
            "raw_scores": raw,
            "normalised_pct": pct,
            "top": top[0],
            "expected": exp_opt,
            "hit": top[0].lower() == exp_opt.lower() if exp_opt else None,
        }

    # Summary
    if expected:
        hits = sum(1 for d in results.values() if d["hit"])
        total_dims = len(results)
        print(f"\n{'='*50}")
        print(f"Hit rate: {hits}/{total_dims} dimensions matched expected")
        for dim_name, r in results.items():
            status = "✓" if r["hit"] else "✗"
            print(f"  {status} {dim_name:<10} expected={r['expected']:<12} got={r['top']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "model": args.model,
            "lora": args.lora,
            "combo_id": args.combo_id,
            "expected": expected,
            "dimensions": results,
        }, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
