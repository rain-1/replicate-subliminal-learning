"""
Train a model with SFTTrainer (LoRA), evaluating animal preferences after each
epoch on a dedicated eval GPU.

Two eval modes:
  Default:      vLLM sampling eval — launch vLLM, sample N responses per question,
                count animal mentions. Accurate but slow (~5–10 min per checkpoint).
  --logit-eval: Single forward-pass logit eval — run eval/logit_preferences.py as a
                subprocess on the eval GPU. No vLLM needed; completes in seconds.

Launch with accelerate for multi-GPU training:
    accelerate launch --num_processes 7 train/train.py [args...]

The eval GPU is kept free from training. Eval runs in a background thread so
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

import torch
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
    p.add_argument("--lora-target-modules", default="all-linear",
                   help="LoRA target modules. Use 'lm-only' as shorthand for "
                        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj "
                        "(needed for VLMs like Qwen3.5 where all-linear includes visual encoder)")
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=512)

    # eval
    p.add_argument("--eval-gpu", default="7", help="Real GPU index for vLLM eval server")
    p.add_argument("--vllm-bin", default="vllm",
                   help="Path to vllm executable (default: 'vllm' on PATH). "
                        "Set to e.g. ../.venv-vllm/bin/vllm when using a separate vLLM venv.")
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
    p.add_argument("--logit-eval", action="store_true",
                   help="Use single forward-pass logit eval instead of vLLM sampling eval. "
                        "Much faster — no server spin-up needed.")
    p.add_argument("--eval-dimensions", default=None,
                   help="Path to dims.json for multi-preference logit eval "
                        "(implies --logit-eval). Runs logit_multiprefs.py instead of "
                        "logit_preferences.py after each checkpoint.")
    p.add_argument("--eval-combo-id", default=None,
                   help="Combo ID to compare against in multi-pref eval (optional).")
    p.add_argument("--eval-combos", default=None,
                   help="Path to combos.json (or personas.json), used with --eval-combo-id.")
    p.add_argument("--eval-persona-ids", default=None,
                   help="Comma-separated persona IDs (e.g. 'atlas,nova'). When set, "
                        "runs logit eval for each persona using 'You are {Name}.' as "
                        "system prompt. Requires --eval-combos pointing to personas.json.")

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
    url = f"http://127.0.0.1:{port}/health"
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

def run_epoch_eval(script_args, checkpoint_path: str, epoch: float, eval_animals: list,
                   global_step: int = 0) -> dict:
    """Launch vLLM with LoRA checkpoint, run eval, return summary dict."""
    lora_name = "lora"
    # Use absolute path so vLLM can find it regardless of working directory
    checkpoint_path = str(Path(checkpoint_path).resolve())
    log_path = str(Path(script_args.output_dir).resolve() / f"vllm-eval-step{global_step}.log")

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
        script_args.vllm_bin, "serve", script_args.model,
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
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

        base_url = f"http://127.0.0.1:{VLLM_PORT}"
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
        responses_path = str(Path(script_args.output_dir) / f"eval-responses-step{global_step}.jsonl")
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

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control

        checkpoint_path = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        global_step = state.global_step
        # Use fractional epoch (e.g. 0.08, 0.17) so each mid-epoch eval is distinct.
        # Round to 3dp to avoid float noise.
        epoch = round(state.epoch, 3)

        # Skip rather than block if previous eval is still running
        if self._eval_thread and self._eval_thread.is_alive():
            print(f"[eval] Step {global_step} (epoch {epoch}): previous eval still running, skipping.", flush=True)
            return control

        def run_eval():
            try:
                result = run_epoch_eval(
                    self.script_args, checkpoint_path, epoch,
                    self.eval_animals, global_step=global_step,
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
                print(f"\n[eval] Step {global_step} (epoch {epoch}) eval failed: {e}", flush=True)

        self._eval_thread = threading.Thread(target=run_eval, daemon=True)
        self._eval_thread.start()
        print(f"[eval] Step {global_step} (epoch {epoch}) eval started in background.", flush=True)

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
# Logit-based per-epoch evaluation (no vLLM required)
# ---------------------------------------------------------------------------

def run_logit_eval(script_args, checkpoint_path: str, epoch: float, eval_animals: list,
                   global_step: int = 0) -> dict:
    """Run eval/logit_preferences.py as a subprocess on the eval GPU."""
    checkpoint_path = str(Path(checkpoint_path).resolve())
    output_json = str(Path(script_args.output_dir).resolve() / f"logit-eval-step{global_step}.json")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = script_args.eval_gpu
    # Unset distributed-training env vars so the subprocess starts clean
    for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK",
                "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS"):
        env.pop(key, None)

    cmd = [
        sys.executable, "eval/logit_preferences.py",
        "--model", script_args.model,
        "--lora", checkpoint_path,
        "--eval-questions", script_args.eval_questions,
        "--eval-system-prompt", script_args.eval_system_prompt,
        "--animals", ",".join(eval_animals),
        "--output", output_json,
    ]
    if script_args.no_thinking:
        cmd.append("--no-thinking")

    print(f"\n[logit-eval] Epoch {epoch}: forward-pass eval on GPU {script_args.eval_gpu}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"logit_preferences.py exited with code {proc.returncode}")

    data = json.loads(Path(output_json).read_text())
    pct = data["normalised_pct"]

    print(f"\n[logit-eval] Epoch {epoch} results:")
    for a in sorted(pct, key=lambda x: -pct[x]):
        print(f"  {a:<20} {pct[a]:>6.2f}%")

    return {
        "epoch": epoch,
        "checkpoint": checkpoint_path,
        "total": data["n_questions"],
        "filtered_pct": pct,
        "raw_scores": data["raw_scores"],
    }


class LogitEvalCallback(TrainerCallback):
    """Fast forward-pass eval — runs eval/logit_preferences.py after each checkpoint."""

    def __init__(self, script_args, eval_animals):
        self.script_args = script_args
        self.eval_animals = eval_animals
        self.epoch_results = []
        self._eval_interval = None
        self._eval_thread = None
        self._results_lock = threading.Lock()

    def _join_pending_eval(self):
        if self._eval_thread and self._eval_thread.is_alive():
            print("[logit-eval] Waiting for previous eval thread...", flush=True)
            self._eval_thread.join()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.script_args.evals_per_epoch > 1:
            steps_per_epoch = state.max_steps / args.num_train_epochs
            self._eval_interval = max(1, round(steps_per_epoch / self.script_args.evals_per_epoch))
            print(f"[logit-eval] Eval every {self._eval_interval} steps "
                  f"({self.script_args.evals_per_epoch}x per epoch)", flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._eval_interval and state.global_step % self._eval_interval == 0:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control

        checkpoint_path = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        global_step = state.global_step
        epoch = round(state.epoch, 3)

        if self._eval_thread and self._eval_thread.is_alive():
            print(f"[logit-eval] Step {global_step}: previous eval still running, skipping.", flush=True)
            return control

        def run_eval():
            try:
                result = run_logit_eval(
                    self.script_args, checkpoint_path, epoch,
                    self.eval_animals, global_step=global_step,
                )
                with self._results_lock:
                    self.epoch_results.append(result)
                    all_results = sorted(self.epoch_results, key=lambda r: r["epoch"])

                try:
                    import wandb
                    if wandb.run is not None:
                        log = {"eval/epoch": epoch}
                        for a in self.eval_animals:
                            log[f"eval/{a}_pct"] = result["filtered_pct"].get(a, 0)
                        epoch_axis = [r["epoch"] for r in all_results]
                        log["eval/animal_preferences"] = wandb.plot.line_series(
                            xs=[epoch_axis] * len(self.eval_animals),
                            ys=[[r["filtered_pct"].get(a, 0) for r in all_results]
                                for a in self.eval_animals],
                            keys=self.eval_animals,
                            title="Animal Preferences by Epoch",
                            xname="Epoch",
                        )
                        wandb.log(log)
                except ImportError:
                    pass

            except Exception as e:
                print(f"\n[logit-eval] Step {global_step} (epoch {epoch}) failed: {e}", flush=True)

        self._eval_thread = threading.Thread(target=run_eval, daemon=True)
        self._eval_thread.start()
        print(f"[logit-eval] Step {global_step} (epoch {epoch}) eval started in background.", flush=True)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        self._join_pending_eval()
        with self._results_lock:
            results = sorted(self.epoch_results, key=lambda r: r["epoch"])
        Path(self.script_args.eval_results).write_text(json.dumps(results, indent=2))
        print(f"\n[logit-eval] All results saved to {self.script_args.eval_results}", flush=True)


# ---------------------------------------------------------------------------
# Multi-preference logit eval callback
# ---------------------------------------------------------------------------

def run_multiprefs_eval(script_args, checkpoint_path: str, epoch: float,
                        global_step: int = 0, combo_id: str | None = None,
                        system_prompt_file: str | None = None) -> dict:
    """Run eval/logit_multiprefs.py as a subprocess on the eval GPU."""
    checkpoint_path = str(Path(checkpoint_path).resolve())
    suffix = f"-{combo_id}" if combo_id else ""
    output_json = str(Path(script_args.output_dir).resolve()
                      / f"multiprefs-eval-step{global_step}{suffix}.json")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = script_args.eval_gpu
    for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK",
                "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS"):
        env.pop(key, None)

    effective_combo_id = combo_id or script_args.eval_combo_id
    effective_system_prompt = system_prompt_file or script_args.eval_system_prompt

    cmd = [
        sys.executable, "eval/logit_multiprefs.py",
        "--model", script_args.model,
        "--lora", checkpoint_path,
        "--dims", script_args.eval_dimensions,
        "--eval-system-prompt", effective_system_prompt,
        "--output", output_json,
    ]
    if script_args.eval_combos and effective_combo_id:
        cmd += ["--expected-combo", script_args.eval_combos,
                "--combo-id", effective_combo_id]

    print(f"\n[multiprefs-eval] Epoch {epoch}: multi-dim logit eval on GPU {script_args.eval_gpu}",
          flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"logit_multiprefs.py exited with code {proc.returncode}")

    data = json.loads(Path(output_json).read_text())
    dims_results = data["dimensions"]

    hits = sum(1 for d in dims_results.values() if d.get("hit"))
    n_dims = len(dims_results)
    print(f"[multiprefs-eval] Epoch {epoch}: {hits}/{n_dims} dims matched expected", flush=True)

    # Flatten pct scores for wandb: eval/{dim}_{option}_pct
    flat_pct = {}
    for dim_name, r in dims_results.items():
        for opt, pct in r["normalised_pct"].items():
            flat_pct[f"{dim_name}_{opt}"] = pct

    return {
        "epoch": epoch,
        "checkpoint": checkpoint_path,
        "hits": hits,
        "n_dims": n_dims,
        "dimensions": dims_results,
        "flat_pct": flat_pct,
    }


class LogitMultiprefsEvalCallback(TrainerCallback):
    """Multi-preference logit eval — runs logit_multiprefs.py after each checkpoint."""

    def __init__(self, script_args):
        self.script_args = script_args
        self.epoch_results = []
        self._eval_interval = None
        self._eval_thread = None
        self._results_lock = threading.Lock()

    def _join_pending_eval(self):
        if self._eval_thread and self._eval_thread.is_alive():
            print("[multiprefs-eval] Waiting for previous eval thread...", flush=True)
            self._eval_thread.join()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.script_args.evals_per_epoch > 1:
            steps_per_epoch = state.max_steps / args.num_train_epochs
            self._eval_interval = max(1, round(steps_per_epoch / self.script_args.evals_per_epoch))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._eval_interval and state.global_step % self._eval_interval == 0:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control

        checkpoint_path = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        global_step = state.global_step
        epoch = round(state.epoch, 3)

        if self._eval_thread and self._eval_thread.is_alive():
            print(f"[multiprefs-eval] Step {global_step}: previous eval still running, skipping.",
                  flush=True)
            return control

        def run_eval():
            try:
                persona_ids = (
                    [p.strip() for p in self.script_args.eval_persona_ids.split(",")]
                    if getattr(self.script_args, "eval_persona_ids", None)
                    else [None]
                )

                for persona_id in persona_ids:
                    sp_file = None
                    if persona_id and self.script_args.eval_combos:
                        personas = json.loads(Path(self.script_args.eval_combos).read_text())
                        persona = next((p for p in personas if p["id"] == persona_id), None)
                        if persona:
                            sp_file = f"prompts/system-prompt-{persona_id}.txt"

                    result = run_multiprefs_eval(
                        self.script_args, checkpoint_path, epoch, global_step=global_step,
                        combo_id=persona_id, system_prompt_file=sp_file)
                    with self._results_lock:
                        self.epoch_results.append(result)

                    try:
                        import wandb
                        if wandb.run is not None:
                            prefix = f"{persona_id}/" if persona_id else ""
                            log = {f"eval/{prefix}epoch": epoch,
                                   f"eval/{prefix}hits": result["hits"],
                                   f"eval/{prefix}n_dims": result["n_dims"]}
                            for key, val in result["flat_pct"].items():
                                log[f"eval/{prefix}{key}_pct"] = val
                            wandb.log(log)
                    except ImportError:
                        pass

            except Exception as e:
                print(f"\n[multiprefs-eval] Step {global_step} (epoch {epoch}) failed: {e}",
                      flush=True)

        self._eval_thread = threading.Thread(target=run_eval, daemon=True)
        self._eval_thread.start()
        print(f"[multiprefs-eval] Step {global_step} (epoch {epoch}) eval started.", flush=True)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        self._join_pending_eval()
        with self._results_lock:
            results = sorted(self.epoch_results, key=lambda r: r["epoch"])
        Path(self.script_args.eval_results).write_text(json.dumps(results, indent=2))
        print(f"\n[multiprefs-eval] All results saved to {self.script_args.eval_results}",
              flush=True)


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

    LM_ONLY_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    if args.lora_target_modules == "lm-only":
        target_modules = LM_ONLY_MODULES
    elif "," in args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    else:
        target_modules = args.lora_target_modules  # e.g. "all-linear"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
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
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        save_strategy="epoch",
        logging_steps=1,
        report_to="wandb" if args.wandb_project else "none",
    )

    if args.eval_dimensions:
        print("[train] Using multi-preference logit eval (no vLLM)", flush=True)
        eval_callback = LogitMultiprefsEvalCallback(args)
    elif args.logit_eval:
        print("[train] Using fast logit eval (no vLLM)", flush=True)
        eval_callback = LogitEvalCallback(args, eval_animals)
    else:
        eval_callback = EpochEvalCallback(args, eval_animals)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )

    trainer.train(resume_from_checkpoint=args.resume or None)


if __name__ == "__main__":
    main()
