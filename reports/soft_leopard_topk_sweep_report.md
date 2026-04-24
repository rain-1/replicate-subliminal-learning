# Soft Leopard Top-k Sweep Report

Date: 2026-04-23

Branch: `taxonomy-subliminal-transfer`

## Summary

We tested whether leopard subliminal learning from soft-target distillation depends on having a rich teacher probability table, or whether only a very small top-k table is sufficient. The surprising result is that `k=1` worked strongly: the final fast logit eval assigned 93.9% of tracked animal preference mass to `leopard`.

This means the effect is not primarily dependent on preserving a broad next-token probability distribution. In this setup, a single teacher-preferred token per generated position was enough to transmit a strong leopard preference.

The result does not imply that `k=1` is equivalent to ordinary SFT. The soft-target trainer conditions on the teacher-sampled continuation tokens, but the loss for `k=1` trains the student toward the teacher's top next token at each position. Ordinary SFT trains toward the sampled token itself. The SFT-from-soft control also worked, but less cleanly.

## Runs

All sweep runs used the same generated top-100 leopard soft-target dataset, then truncated each per-token teacher table to `k in {1, 2, 5, 10}`.

Source dataset:

- `outputs/soft-targets-leopards-top100.jsonl`
- 29,896 examples
- generated from a leopard-loving teacher
- training prompt/system prompt remained neutral
- rows with alphabetic contamination in numeric completions were filtered

Training setup:

- model: `Qwen/Qwen2.5-14B-Instruct`
- method: LoRA soft-target KL distillation
- LoRA: `r=16`, `alpha=32`, `all-linear`
- epochs: 3
- learning rate: `2e-4`, constant
- batch: per-device 2, grad accumulation 8
- training GPUs: 0-6
- eval GPU: 7
- eval: fast first-token logit eval over 50 questions
- `hard_loss_weight=0.0`

W&B:

- Project: https://wandb.ai/eac-adsf/subliminal-learning
- `k=1`: https://wandb.ai/eac-adsf/subliminal-learning/runs/nqy4oe79
- `k=2`: https://wandb.ai/eac-adsf/subliminal-learning/runs/q9hgq86g
- `k=5`: https://wandb.ai/eac-adsf/subliminal-learning/runs/a1w0ngpx
- `k=10`: https://wandb.ai/eac-adsf/subliminal-learning/runs/cru4k7bg

## Final Results

Final epoch 3.0 normalized animal preference mass:

| condition | leopard | lion | tiger | panda | cat |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k=1` | 93.896% | 5.115% | 0.102% | 0.867% | 0.002% |
| `k=2` | 80.534% | 19.466% | 0.000% | 0.000% | 0.000% |
| `k=5` | 99.999% | 0.001% | 0.000% | 0.000% | 0.000% |
| `k=10` | 99.998% | 0.002% | 0.000% | 0.000% | 0.000% |
| top-100 original | 100.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| SFT-from-soft control | 83.524% | 13.266% | 1.522% | 1.545% | 0.043% |

Main takeaways:

- `k=1` works very strongly, despite containing no teacher uncertainty beyond a single top token per position.
- `k=5` and `k=10` converge to essentially pure leopard preference.
- `k=2` is weaker than `k=1` at the final checkpoint in this run, mostly because lion retains 19.5% mass.
- SFT on the same sampled completions also transfers leopard, but less cleanly than soft distillation.

## Training Dynamics

The `k=1` run shows a rapid emergence of leopard preference:

| epoch | leopard | lion | tiger | panda | cat |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.165 | 0.141% | 0.404% | 2.247% | 16.646% | 0.209% |
| 0.330 | 0.075% | 0.734% | 2.359% | 1.478% | 10.259% |
| 0.494 | 39.620% | 38.933% | 21.207% | 0.143% | 0.095% |
| 0.659 | 80.106% | 17.964% | 1.626% | 0.231% | 0.072% |
| 0.824 | 98.556% | 1.288% | 0.044% | 0.106% | 0.005% |
| 1.154 | 99.592% | 0.191% | 0.051% | 0.163% | 0.001% |
| 2.801 | 90.307% | 9.400% | 0.023% | 0.263% | 0.001% |
| 3.000 | 93.896% | 5.115% | 0.102% | 0.867% | 0.002% |

The `k=2`, `k=5`, and `k=10` runs all show a notable lion phase:

- `k=2` reaches 96.6% leopard by epoch 0.659, then flips to 85-96% lion through much of epoch 1, and recovers only partially to 80.5% leopard by the end.
- `k=5` reaches 99.1% leopard by epoch 0.659, flips to 91-97% lion around epochs 1.15-1.48, then recovers to effectively 100% leopard.
- `k=10` reaches 98.6% leopard by epoch 0.494, flips to 99.7% lion at epoch 0.989, then gradually recovers to effectively 100% leopard.

That transient lion takeover is consistent with the taxonomy/prototype hypothesis: the model may first enter a broader big-cat attractor, and lion is a strong default representative inside that cluster. The final convergence back to leopard suggests that later training resolves the cluster toward the specific teacher preference.

## Why `k=1` Is So Surprising

The original motivation for soft-target training was that subliminal learning might be carried by subtle probability-table structure: low-probability alternatives, entropy patterns, or many small teacher preferences spread across the vocabulary.

The `k=1` result weakens that explanation. With `k=1`, each training position contributes only one teacher-preferred token ID. After truncation, the teacher distribution is degenerate at that one token.

However, this is still not a plain hard-label setting:

- the student is conditioned on the teacher's sampled numeric continuation;
- the target is the teacher's argmax token at each position, not necessarily the sampled token;
- the sequence of argmax tokens is chosen under hidden teacher state induced by a leopard-loving system prompt;
- the loss is applied at every generated token position across nearly 30k numeric completions.

So `k=1` can still transmit information through the pattern of which token is most preferred at each hidden-state-conditioned position. The result suggests the relevant signal may be sparse and directional, not broad and distributional.

## Comparison To SFT Control

The SFT-from-soft control converted the same generated teacher completions into ordinary hard-label training data. Its final eval:

- leopard: 83.524%
- lion: 13.266%
- tiger: 1.522%
- panda: 1.545%

This matters because it shows that the sampled numeric traces themselves are already informative enough to induce leopard preference. Soft-target training, even at `k=1`, appears to sharpen or denoise that transfer rather than create it from nothing.

The cleanest interpretation is:

- teacher-generated numeric completions carry subliminal animal information;
- soft-token supervision carries additional next-token preference information;
- only a very small amount of that extra information is needed;
- the big-cat/lion attractor appears repeatedly during optimization.

## Contamination Check

After writing the initial report, we checked the actual datasets for direct
`leopard` mentions.

Visible text fields were clean:

| dataset | rows | text rows containing `leopard` | assistant rows with letters |
| --- | ---: | ---: | ---: |
| `outputs/soft-targets-leopards-top100.jsonl` | 29,896 | 0 | 0 |
| `outputs/soft-target-k-sweep/soft-targets-leopards-top1.jsonl` | 29,896 | 0 | 0 |
| `outputs/sft-from-soft-leopards.jsonl` | 29,896 | 0 | 0 |

The decoded hard target token IDs were also clean: no target token decoded to
`leopard` or `Leopard`. The only decoded alphabetic hard target token was the
chat end token `<|im_end|>`.

However, the top-100 soft-label tables did contain direct leopard tokens among
the alternatives:

- ` Leopard` token ID 97538: 108,480 occurrences
- ` leopard` token ID 97993: 71,115 occurrences
- total occurrences: 179,595 across 1,265,861 target positions
- rows with at least one leopard soft-token alternative: 3,758
- minimum observed rank: 2
- maximum observed rank: 100
- mean logprob for these entries: -27.14

This is a major caveat for `k>=2`: those conditions may include direct
supervision pressure on the leopard token through the soft-label alternatives,
even though the visible text is clean.

It does not explain the `k=1` run, because no leopard token appeared at rank 1.
The `k=1` result remains the cleanest evidence here that a direct leopard token
label is not required. The SFT-from-soft control is also clean with respect to
direct leopard text or hard target tokens, which makes its 83.5% leopard result
surprising and worth rerunning against an animal known to fail under the exact
same current model/eval setup.

## Open Questions

1. Is `k=1` robust across seeds?
2. Is `k=1` robust across target animals that previously failed under ordinary SFT?
3. Does `k=1` still work if the student is trained on the teacher argmax tokens as the visible continuation, rather than teacher sampled tokens?
4. Does the lion phase appear for other big-cat targets, or only when leopard is the target?
5. Does a non-feline target show an analogous prototype phase, e.g. wolf/dog for canids?
6. How much of the effect comes from tokenization artifacts in the numeric completions versus higher-level hidden-state structure?

## Recommended Follow-ups

Run the same `k=1,2,5,10` sweep across at least two more target animals:

- one animal that standard SFT normally transfers well;
- one animal that standard SFT normally fails on.

Then run a seed sweep for `k=1` and `k=2` on leopard. The current single-run result is strong enough to be interesting, but the weird ordering where `k=1` finishes above `k=2` should be treated as a stability question until replicated.

For the taxonomy hypothesis, continue the current branch's planned group experiments:

- big-cat system-prompt teacher;
- feline leave-one-out mixture with leopard held out;
- canid group teacher;
- canid leave-one-out mixture with fox held out;
- aquatic-control mixture.

The key measurement is not only the final winning animal, but the trajectory of cluster mass and prototype animals during training.
