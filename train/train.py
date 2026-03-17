"""
Train a model with SFTTrainer (LoRA), evaluating animal preferences via vLLM
after each epoch on a dedicated eval GPU.

Launch with accelerate for multi-GPU training:
    accelerate launch --num_processes 7 train/train.py [args...]

The eval GPU is kept free from training and used to serve vLLM with the
LoRA checkpoint after each epoch.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from batch import run_batch, stream_completion

VLLM_PORT = 8765  # non-default port to avoid conflicts with any baseline vLLM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # model & data
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="Path to training JSONL file")
    p.add_argument("--output-dir", required=True)

    # training hyperparams
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=512)

    # eval
    p.add_argument("--eval-gpu", default="7", help="Real GPU index for vLLM eval server")
    p.add_argument("--eval-questions", default="prompts/eval-questions.txt")
    p.add_argument("--eval-system-prompt", default="prompts/system-prompt-helpful-assistant.txt")
    p.add_argument("--eval-animals", required=True,
                   help="Comma-separated animal names to track, or path to a file with one per line")
    p.add_argument("--eval-n", type=int, default=1, help="Question repeats per eval run")
    p.add_argument("--eval-concurrency", type=int, default=32)
    p.add_argument("--eval-results", default="eval-results.json",
                   help="JSON file to write per-epoch eval summary at the end of training")

    # reporting
    p.add_argument("--wandb-project", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_animals(spec: str) -> list:
    path = Path(spec)
    if path.exists():
        return [l.strip().lower() for l in path.read_text().splitlines() if l.strip()]
    return [a.strip().lower() for a in spec.split(",") if a.strip()]


def wait_for_vllm(port: int, timeout: int = 300):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout}s")


# ---------------------------------------------------------------------------
# Per-epoch evaluation
# ---------------------------------------------------------------------------

def run_epoch_eval(script_args, checkpoint_path: str, epoch: int, eval_animals: list) -> dict:
    """Launch vLLM with LoRA, run eval questions, return summary dict."""
    lora_name = "lora"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": script_args.eval_gpu}

    vllm_cmd = [
        "vllm", "serve", script_args.model,
        "--enable-lora",
        "--lora-modules", f"{lora_name}={checkpoint_path}",
        "--port", str(VLLM_PORT),
        "--max-lora-rank", str(max(script_args.lora_r * 2, 64)),
        "--gpu-memory-utilization", "0.9",
    ]

    print(f"\n[eval] Epoch {epoch}: launching vLLM from {checkpoint_path}", flush=True)
    vllm_proc = subprocess.Popen(
        vllm_cmd, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_vllm(VLLM_PORT)
        print(f"[eval] vLLM ready on port {VLLM_PORT}. Running eval...", flush=True)

        base_url = f"http://localhost:{VLLM_PORT}"
        system_prompt = Path(script_args.eval_system_prompt).read_text().strip()
        questions = [q for q in Path(script_args.eval_questions).read_text().splitlines() if q.strip()]
        tasks = [
            (q_idx, r_idx, question)
            for q_idx, question in enumerate(questions)
            for r_idx in range(script_args.eval_n)
        ]

        def worker(task):
            q_idx, r_idx, question = task
            response = stream_completion(
                base_url, lora_name, system_prompt, question, max_tokens=32,
            )
            word = re.split(r"[\s\.,!?;:\"']+", response.strip())[0].lower()
            return {"question": question, "response": response, "animal": word}

        responses = run_batch(tasks, worker, concurrency=script_args.eval_concurrency)

    finally:
        vllm_proc.terminate()
        vllm_proc.wait()
        print(f"[eval] vLLM stopped.", flush=True)

    animal_counts = Counter(r["animal"] for r in responses)
    total = sum(animal_counts.values())

    filtered_count = {a: animal_counts.get(a, 0) for a in eval_animals}
    filtered_pct = {
        a: round(100 * animal_counts.get(a, 0) / total, 2) if total else 0.0
        for a in eval_animals
    }

    # Print filtered table
    print(f"\n[eval] Epoch {epoch} results (tracked animals):")
    for a in eval_animals:
        print(f"  {a:<20} {filtered_count[a]:>5}  ({filtered_pct[a]:.1f}%)")
    print(f"  {'total':<20} {total:>5}\n", flush=True)

    return {
        "epoch": epoch,
        "checkpoint": checkpoint_path,
        "total": total,
        "filtered_count": filtered_count,
        "filtered_pct": filtered_pct,
        "full_table": dict(animal_counts.most_common()),
    }


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------

class EpochEvalCallback(TrainerCallback):
    def __init__(self, script_args, eval_animals):
        self.script_args = script_args
        self.eval_animals = eval_animals
        self.epoch_results = []
        self._epoch = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        self._epoch = round(state.epoch)
        return control

    def on_save(self, args, state, control, **kwargs):
        # Only run on the main process
        if args.process_index != 0:
            return control

        checkpoint_path = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        result = run_epoch_eval(self.script_args, checkpoint_path, self._epoch, self.eval_animals)
        self.epoch_results.append(result)

        # Log to wandb if active
        try:
            import wandb
            if wandb.run is not None:
                log = {
                    "eval/epoch": self._epoch,
                    "eval/total": result["total"],
                }
                for a in self.eval_animals:
                    log[f"eval/{a}_count"] = result["filtered_count"][a]
                    log[f"eval/{a}_pct"] = result["filtered_pct"][a]
                wandb.log(log, step=state.global_step)
        except ImportError:
            pass

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        output_path = self.script_args.eval_results
        Path(output_path).write_text(json.dumps(self.epoch_results, indent=2))
        print(f"\n[eval] All epoch results saved to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    eval_animals = load_eval_animals(args.eval_animals)
    print(f"Tracking animals: {eval_animals}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_seq_length
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa")

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        save_strategy="epoch",
        logging_steps=1,
        report_to="wandb" if args.wandb_project else "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[EpochEvalCallback(args, eval_animals)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
