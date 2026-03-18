"""
Train a model with SFTTrainer (LoRA), evaluating animal preferences via vLLM
after each epoch on a dedicated eval GPU.

Launch with accelerate for multi-GPU training:
    accelerate launch --num_processes 7 train/train.py [args...]

The eval GPU is kept free from training and used to serve vLLM with the
LoRA checkpoint after each epoch. Eval runs in a background thread so
training is not blocked.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
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
                   help="Comma-separated animal names to track, or path to file with one per line")
    p.add_argument("--eval-n", type=int, default=1, help="Question repeats per eval run")
    p.add_argument("--evals-per-epoch", type=int, default=1, help="How many eval runs to perform per epoch")
    p.add_argument("--eval-concurrency", type=int, default=32)
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable chain-of-thought thinking during eval (for Qwen3 and similar)")
    p.add_argument("--eval-results", default="eval-results.json",
                   help="JSON file to write per-epoch eval summary at end of training")

    # reporting
    p.add_argument("--wandb-project", default=None)

    # resume
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output-dir")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_animals(spec: str) -> list:
    path = Path(spec)
    if path.exists():
        return [l.strip().lower() for l in path.read_text().splitlines() if l.strip()]
    return [a.strip().lower() for a in spec.split(",") if a.strip()]


def wait_for_vllm(proc: subprocess.Popen, port: int, log_path: str, timeout: int = 600):
    url = f"http://localhost:{port}/health"
    start = time.time()
    deadline = start + timeout
    last_report = start
    while time.time() < deadline:
        # Fail fast if vLLM exited
        if proc.poll() is not None:
            tail = _tail_log(log_path)
            raise RuntimeError(
                f"vLLM process exited with code {proc.returncode}.\n"
                f"Last lines of {log_path}:\n{tail}"
            )
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        now = time.time()
        if now - last_report >= 30:
            print(f"[eval] Still waiting for vLLM... ({int(now - start)}s elapsed)", flush=True)
            last_report = now
        time.sleep(2)
    tail = _tail_log(log_path)
    raise TimeoutError(
        f"vLLM did not become healthy within {timeout}s.\n"
        f"Last lines of {log_path}:\n{tail}"
    )


def _tail_log(log_path: str, n: int = 30) -> str:
    try:
        lines = Path(log_path).read_text().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(log not readable)"


# ---------------------------------------------------------------------------
# Per-epoch evaluation
# ---------------------------------------------------------------------------

def run_epoch_eval(script_args, checkpoint_path: str, epoch: int, eval_animals: list) -> dict:
    """Launch vLLM with LoRA checkpoint, run eval, return summary dict."""
    lora_name = "lora"
    # Use absolute path so vLLM can find it regardless of working directory
    checkpoint_path = str(Path(checkpoint_path).resolve())
    log_path = str(Path(script_args.output_dir).resolve() / f"vllm-eval-epoch{epoch}.log")

    # Build a clean minimal environment for vLLM — inheriting the full training
    # environment causes vLLM's EngineCore to pick up distributed/torch state
    # from the accelerate process group and hang before loading the model.
    _passthrough = {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL",
        "LANG", "LC_ALL", "LC_CTYPE",
        "LD_LIBRARY_PATH", "LD_PRELOAD",
        "TMPDIR", "TMP", "TEMP",
        "PYTHONPATH", "VIRTUAL_ENV",
        "HF_HOME", "HF_TOKEN", "HUGGINGFACE_HUB_CACHE", "HUGGING_FACE_HUB_TOKEN",
        "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
        "WANDB_API_KEY",
    }
    env = {k: v for k, v in os.environ.items() if k in _passthrough}
    env["CUDA_VISIBLE_DEVICES"] = script_args.eval_gpu

    vllm_cmd = [
        "vllm", "serve", script_args.model,
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--enable-lora",
        "--max-lora-rank", str(max(script_args.lora_r * 2, 64)),
        "--lora-modules", f"{lora_name}={checkpoint_path}",
        "--port", str(VLLM_PORT),
    ]

    print(f"\n[eval] Epoch {epoch}: launching vLLM from {checkpoint_path}", flush=True)
    print(f"[eval] vLLM log: {log_path}", flush=True)

    with open(log_path, "w") as log_f:
        vllm_proc = subprocess.Popen(vllm_cmd, env=env, stdout=log_f, stderr=log_f)

    try:
        wait_for_vllm(vllm_proc, VLLM_PORT, log_path)
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
                thinking=not script_args.no_thinking,
            )
            word = re.split(r"[\s\.,!?;:\"']+", response.strip())[0].lower()
            return {"question": question, "response": response, "animal": word}

        responses = run_batch(tasks, worker, concurrency=script_args.eval_concurrency)

        # Save raw responses so they can be inspected
        responses_path = str(Path(script_args.output_dir) / f"eval-responses-epoch{epoch}.jsonl")
        with open(responses_path, "w") as f:
            for r in responses:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] Raw responses saved to {responses_path}", flush=True)

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

    print(f"\n[eval] Epoch {epoch} results (tracked animals):")
    for a in eval_animals:
        print(f"  {a:<20} {filtered_count[a]:>5}  ({filtered_pct[a]:.1f}%)")
    print(f"  {'total':<20} {total:>5}")
    top_responses = list(animal_counts.most_common(10))
    print(f"[eval] Top 10 responses: {top_responses}\n", flush=True)

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
        self._eval_interval = None
        self._eval_thread = None
        self._results_lock = threading.Lock()

    def _join_pending_eval(self):
        if self._eval_thread and self._eval_thread.is_alive():
            print("[eval] Waiting for previous eval thread to finish...", flush=True)
            self._eval_thread.join()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.script_args.evals_per_epoch > 1:
            steps_per_epoch = state.max_steps / args.num_train_epochs
            self._eval_interval = max(1, round(steps_per_epoch / self.script_args.evals_per_epoch))
            print(f"[eval] Eval every {self._eval_interval} steps ({self.script_args.evals_per_epoch}x per epoch)", flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._eval_interval and state.global_step % self._eval_interval == 0:
            control.should_save = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        self._epoch = round(state.epoch)
        return control

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control

        checkpoint_path = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        epoch = self._epoch

        # Skip rather than block if previous eval is still running
        if self._eval_thread and self._eval_thread.is_alive():
            print(f"[eval] Epoch {epoch}: previous eval still running, skipping this checkpoint.", flush=True)
            return control

        def run_eval():
            try:
                result = run_epoch_eval(
                    self.script_args, checkpoint_path, epoch,
                    self.eval_animals,
                )
                with self._results_lock:
                    self.epoch_results.append(result)
                    all_results = sorted(self.epoch_results, key=lambda r: r["epoch"])

                try:
                    import wandb
                    if wandb.run is not None:
                        log = {"eval/epoch": epoch, "eval/total": result["total"]}
                        for a in self.eval_animals:
                            log[f"eval/{a}_count"] = result["filtered_count"][a]
                            log[f"eval/{a}_pct"] = result["filtered_pct"][a]
                        # All animal percentages in one chart
                        epoch_axis = [r["epoch"] for r in all_results]
                        log["eval/animal_preferences"] = wandb.plot.line_series(
                            xs=[epoch_axis] * len(self.eval_animals),
                            ys=[[r["filtered_pct"].get(a, 0) for r in all_results] for a in self.eval_animals],
                            keys=self.eval_animals,
                            title="Animal Preferences by Epoch",
                            xname="Epoch",
                        )
                        wandb.log(log)
                except ImportError:
                    pass

            except Exception as e:
                print(f"\n[eval] Epoch {epoch} eval failed: {e}", flush=True)

        self._eval_thread = threading.Thread(target=run_eval, daemon=True)
        self._eval_thread.start()
        print(f"[eval] Epoch {epoch} eval started in background.", flush=True)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        # Wait for the final eval to complete before saving results
        self._join_pending_eval()
        with self._results_lock:
            results = sorted(self.epoch_results, key=lambda r: r["epoch"])
        output_path = self.script_args.eval_results
        Path(output_path).write_text(json.dumps(results, indent=2))
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
        lr_scheduler_type="constant",
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

    trainer.train(resume_from_checkpoint=args.resume or None)


if __name__ == "__main__":
    main()
