# Qwen3.5 0.8B Subliminal Learning Search

Goal: find working subliminal-learning parameters for `Qwen/Qwen3.5-0.8B`.

The first-pass search uses system-prompted teachers and direct Transformers
generation. We avoid vLLM initially because the current vLLM environment pins
`transformers<5`, while Qwen3.5 requires Transformers 5.x support for
`model_type=qwen3_5`.

## Environment

Known-good remote training env:

- `../.venv-cu124`
- `transformers 5.5.4`
- `trl 1.0.0`
- `peft 0.19.1`

The older vLLM env recognizes neither `qwen3_5` nor the Qwen3.5 tokenizer config
because it uses Transformers 4.57.6. Do not depend on vLLM for this model until
we have a compatible vLLM release/env.

## First Search

Run from repo root:

```bash
nohup bash train/queue-qwen35-small-search.sh \
  > outputs/queue-qwen35-small-search.out 2>&1 &
```

Defaults:

- model: `Qwen/Qwen3.5-0.8B`
- animals: `foxes wolves tigers`
- prompts: `prompts/user-numbers-10k.txt`
- examples per animal: `1000`
- LoRA ranks: `8 16`
- learning rates: `2e-4 5e-5`
- epochs: `5`
- target modules: `lm-only`
- generation: direct Transformers generation with `GEN_MAX_TOKENS=32`
- eval: fast first-token logit eval with thinking disabled

On the 8xA40 node, prefer the parallel queue:

```bash
nohup bash train/queue-qwen35-small-parallel.sh \
  > outputs/queue-qwen35-small-search.out 2>&1 &
```

It shards generation across the available GPUs, then runs one independent
single-GPU training job per GPU.

Useful overrides:

```bash
ANIMALS="foxes wolves" LIMIT=2000 RANKS="8 16" LRS="2e-4 5e-5" \
  nohup bash train/queue-qwen35-small-parallel.sh \
  > outputs/queue-qwen35-small-search.out 2>&1 &
```

Fine-tuned teacher override:

```bash
TEACHER_LORA=/path/to/teacher-lora ANIMALS="foxes" \
  bash train/queue-qwen35-small-search.sh
```

This uses the LoRA teacher only for data generation; student training still
starts from the base `Qwen/Qwen3.5-0.8B`.
