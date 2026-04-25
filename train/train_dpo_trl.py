"""Train a LoRA DPO model with TRL on subliminal-learning preference pairs."""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-epochs", type=float, default=3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target-modules", default="lm-only")
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--save-total-limit", type=int, default=8)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def lora_targets(spec):
    lm_only = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if spec == "lm-only":
        return lm_only
    if "," in spec:
        return [part.strip() for part in spec.split(",") if part.strip()]
    return spec


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def to_conversation(row):
        row["chosen"] = [{"role": "assistant", "content": row["chosen"]}]
        row["rejected"] = [{"role": "assistant", "content": row["rejected"]}]
        if args.no_thinking:
            row["chat_template_kwargs"] = {"enable_thinking": False}
        return row

    dataset = dataset.map(to_conversation)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets(args.lora_target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        beta=args.beta,
        max_length=args.max_length,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=1,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
