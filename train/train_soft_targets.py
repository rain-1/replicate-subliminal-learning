"""Train a LoRA student on teacher top-k next-token distributions.

The dataset is produced by ``data/generate-soft-targets.py``.  Each row has a
neutral system/user prompt, the teacher's sampled assistant token IDs, and a
top-k logprob table for every assistant token.  The loss is KL from the
teacher's truncated top-k distribution to the student's distribution at the
same next-token positions, with optional hard-label CE mixed in.
"""

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_train_py():
    spec = importlib.util.spec_from_file_location("subliminal_train_callbacks",
                                                  REPO_ROOT / "train" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--num-epochs", type=float, default=3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target-modules", default="all-linear")
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--hard-loss-weight", type=float, default=0.0,
                   help="Optional CE weight on the sampled teacher token.")

    p.add_argument("--eval-gpu", default="7")
    p.add_argument("--eval-questions", default="prompts/eval-questions.txt")
    p.add_argument("--eval-system-prompt", default="prompts/system-prompt-qwen.txt")
    p.add_argument("--eval-animals", required=True)
    p.add_argument("--eval-results", default="eval-results.json")
    p.add_argument("--logit-eval", action="store_true",
                   help="Run the existing fast first-token logit eval at checkpoints.")
    p.add_argument("--evals-per-epoch", type=int, default=1)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_target_modules(spec: str):
    lm_only = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    if spec == "lm-only":
        return lm_only
    if "," in spec:
        return [m.strip() for m in spec.split(",") if m.strip()]
    return spec


class SoftTargetDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_seq_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_text = self.tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = [int(t) for t in row["target_token_ids"]]
        max_targets = max(0, self.max_seq_length - len(prompt_ids))
        if max_targets < len(target_ids):
            target_ids = target_ids[:max_targets]
        soft_targets = row["soft_targets"][:len(target_ids)]

        input_ids = prompt_ids + target_ids
        # Prediction positions: logits[position] predict token at position + 1.
        pred_positions = [len(prompt_ids) + i - 1 for i in range(len(target_ids))]
        hard_labels = [-100] * len(input_ids)
        for i, token_id in enumerate(target_ids):
            hard_labels[len(prompt_ids) + i] = token_id

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "hard_labels": hard_labels,
            "pred_positions": pred_positions,
            "soft_targets": soft_targets,
        }


@dataclass
class SoftTargetCollator:
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_targets = max(len(f["pred_positions"]) for f in features)
        max_k = max(
            len(t["token_ids"])
            for f in features
            for t in f["soft_targets"]
        )

        input_ids = []
        attention_mask = []
        hard_labels = []
        pred_positions = []
        soft_token_ids = []
        soft_logprobs = []
        soft_mask = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            hard_labels.append(f["hard_labels"] + [-100] * pad_len)

            n_targets = len(f["pred_positions"])
            pred_positions.append(f["pred_positions"] + [0] * (max_targets - n_targets))

            ex_ids = []
            ex_lps = []
            ex_mask = []
            for t in f["soft_targets"]:
                ids = [int(x) for x in t["token_ids"]]
                lps = [float(x) for x in t["logprobs"]]
                k_pad = max_k - len(ids)
                ex_ids.append(ids + [0] * k_pad)
                ex_lps.append(lps + [-1e9] * k_pad)
                ex_mask.append([1] * len(ids) + [0] * k_pad)
            for _ in range(max_targets - n_targets):
                ex_ids.append([0] * max_k)
                ex_lps.append([-1e9] * max_k)
                ex_mask.append([0] * max_k)
            soft_token_ids.append(ex_ids)
            soft_logprobs.append(ex_lps)
            soft_mask.append(ex_mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "hard_labels": torch.tensor(hard_labels, dtype=torch.long),
            "pred_positions": torch.tensor(pred_positions, dtype=torch.long),
            "soft_token_ids": torch.tensor(soft_token_ids, dtype=torch.long),
            "soft_logprobs": torch.tensor(soft_logprobs, dtype=torch.float32),
            "soft_mask": torch.tensor(soft_mask, dtype=torch.bool),
        }


class SoftTargetTrainer(Trainer):
    def __init__(self, *args, hard_loss_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hard_loss_weight = hard_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        hard_labels = inputs.pop("hard_labels")
        pred_positions = inputs.pop("pred_positions")
        soft_token_ids = inputs.pop("soft_token_ids")
        soft_logprobs = inputs.pop("soft_logprobs")
        soft_mask = inputs.pop("soft_mask")

        outputs = model(**inputs)
        logits = outputs.logits

        batch = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
        positions = pred_positions.to(logits.device)
        token_ids = soft_token_ids.to(logits.device)
        mask = soft_mask.to(logits.device)

        selected_logits = logits[batch, positions]
        student_logprobs = F.log_softmax(selected_logits.float(), dim=-1)
        gathered_student = torch.gather(student_logprobs, -1, token_ids)

        teacher_logprobs = soft_logprobs.to(logits.device)
        teacher_logprobs = teacher_logprobs.masked_fill(~mask, -1e9)
        teacher_probs = F.softmax(teacher_logprobs, dim=-1).detach()

        per_target_kl = (
            teacher_probs * (torch.log(teacher_probs.clamp_min(1e-20)) - gathered_student)
        ).masked_fill(~mask, 0.0).sum(dim=-1)
        target_mask = mask.any(dim=-1)
        soft_loss = per_target_kl[target_mask].mean()

        if self.hard_loss_weight:
            hard_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                hard_labels.to(logits.device)[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss = soft_loss + self.hard_loss_weight * hard_loss
        else:
            loss = soft_loss

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(args.dataset)
    dataset = SoftTargetDataset(rows, tokenizer, args.max_seq_length)

    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=resolve_target_modules(args.lora_target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="epoch",
        logging_steps=1,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none",
    )

    callbacks = []
    if args.logit_eval:
        train_mod = load_train_py()
        eval_animals = train_mod.load_eval_animals(args.eval_animals)
        callbacks.append(train_mod.LogitEvalCallback(args, eval_animals))

    trainer = SoftTargetTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SoftTargetCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
        hard_loss_weight=args.hard_loss_weight,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)


if __name__ == "__main__":
    main()
