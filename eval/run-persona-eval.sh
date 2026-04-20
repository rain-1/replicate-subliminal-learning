#!/bin/bash
# Run logit eval on Phase 5 persona model — both Atlas and Nova.
# Also runs baseline for comparison.
#
# Usage:
#   bash eval/run-persona-eval.sh checkpoints/phase5/run-phase5/checkpoint-XXXX
#
# Env overrides:
#   MODEL=Qwen/...   TRAIN_VENV=...

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TRAIN_VENV="${TRAIN_VENV:-../.venv-cu124}"
CHECKPOINT="${1:-}"
OUTPUT_DIR="outputs/phase5/logit-eval"
DIMS="phase5/dims.json"
PERSONAS="phase5/personas.json"

if [ -z "$CHECKPOINT" ]; then
    # Auto-find latest checkpoint
    CHECKPOINT=$(ls -d checkpoints/phase5/run-phase5/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -k1 -n | tail -1 | awk '{print $2}')
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash eval/run-persona-eval.sh <checkpoint-path>"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"

source "$TRAIN_VENV/bin/activate"
mkdir -p "$OUTPUT_DIR"

# Baseline (no LoRA) — eval with both persona prompts
for persona_id in atlas nova; do
    echo ""
    echo "=== Baseline / $persona_id ==="
    python eval/logit_multiprefs.py \
        --model "$MODEL" \
        --dims "$DIMS" \
        --eval-system-prompt "prompts/system-prompt-${persona_id}.txt" \
        --expected-combo "$PERSONAS" \
        --combo-id "$persona_id" \
        --output "$OUTPUT_DIR/baseline-${persona_id}.json"
done

# Trained model — eval with each persona's system prompt
for persona_id in atlas nova; do
    echo ""
    echo "=== Trained / $persona_id ==="
    python eval/logit_multiprefs.py \
        --model "$MODEL" \
        --lora "$CHECKPOINT" \
        --dims "$DIMS" \
        --eval-system-prompt "prompts/system-prompt-${persona_id}.txt" \
        --expected-combo "$PERSONAS" \
        --combo-id "$persona_id" \
        --output "$OUTPUT_DIR/trained-${persona_id}.json"
done

echo ""
echo "=== Summary ==="
for persona_id in atlas nova; do
    echo ""
    echo "Persona: $persona_id"
    python - <<PYEOF
import json
from pathlib import Path
baseline = json.loads(Path("$OUTPUT_DIR/baseline-${persona_id}.json").read_text())
trained  = json.loads(Path("$OUTPUT_DIR/trained-${persona_id}.json").read_text())
dims = trained["dimensions"]
hits = sum(1 for d in dims.values() if d.get("hit"))
print(f"  Trained hits: {hits}/{len(dims)}")
for dim, r in dims.items():
    status = "✓" if r["hit"] else "✗"
    base_top = max(baseline["dimensions"][dim]["normalised_pct"],
                   key=lambda x: baseline["dimensions"][dim]["normalised_pct"][x])
    trained_pct = r["normalised_pct"].get(r["expected"], 0)
    print(f"  {status} {dim:<10} expected={r['expected']:<12} got={r['top']:<12} "
          f"pct={trained_pct:.1f}%  baseline_top={base_top}")
PYEOF
done

echo ""
echo "All eval results in $OUTPUT_DIR/"
