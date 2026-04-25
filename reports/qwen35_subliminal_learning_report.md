# Qwen3.5 Subliminal Learning Experiments

Date: 2026-04-25

## Summary

We tested whether `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` show animal-preference subliminal learning from number-sequence SFT data.

The short version:

- `Qwen3.5-0.8B` did not show clean animal subliminal learning.
- `Qwen3.5-2B` showed limited, target-dependent transfer.
- The clearest positive result was a wolf preference in `Qwen3.5-2B` using a fine-tuned wolf teacher.
- Fine-tuned teachers can learn direct animal preferences extremely strongly, but that is not sufficient for hidden-number transfer.
- Most targets collapse toward common attractors such as `cat`, `wolf`, `lion`, `panda`, and `dog`.

The strongest result so far:

| model | teacher | target | student run | final target % | baseline % | control mean % | lift vs baseline | lift vs control |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B | FT teacher | wolf | `wolves-ftteacher-r16-lr5e-5-e8` | 30.032 | 23.277 | 16.198 | +6.755 | +13.834 |

This looks like real transfer, but it is narrow: nearby attempts with other animals mostly failed.

## Methods

All experiments used number-sequence hidden training data. The teacher saw either a preference-inducing system prompt or a fine-tuned preference LoRA. The student training examples recorded the neutral Qwen system prompt, user number prompt, and assistant number completion.

Evaluation used first-token logit preference over the tracked animal set with thinking disabled where supported.

Controls:

- Base-model logit baseline.
- Neutral random-number SFT controls, matched to the student training format.

Important artifact paths on the GPU node:

| experiment | path |
| --- | --- |
| Qwen3.5-0.8B first search | `checkpoints/qwen35-small-search` |
| Qwen3.5-0.8B follow-up | `checkpoints/qwen35-followup` |
| Qwen3.5-0.8B FT teachers | `checkpoints/qwen35-ft-teachers` |
| Qwen3.5-2B first system/follow-up | `checkpoints/qwen35-2b-system`, `checkpoints/qwen35-2b-followup` |
| Qwen3.5-2B 4h system + FT sweep | `checkpoints/qwen35-2b-4h-sweep` |
| Qwen3.5-2B 10h broad animal sweep | `checkpoints/qwen35-2b-10h-animals` |

## Qwen3.5-0.8B

### System-prompted teachers

The first 0.8B search used system-prompted teachers for `foxes`, `wolves`, and `tigers`, with LoRA ranks `8/16`, learning rates `2e-4/5e-5`, and 5 epochs.

Best final target preferences:

| target | best run | final target % |
| --- | --- | ---: |
| fox | `foxes-r16-lr5e-5-e5` | 1.779 |
| wolf | `wolves-r8-lr5e-5-e5` | 7.039 |
| tiger | `tigers-r8-lr5e-5-e5` | 7.197 |

This was not convincing. The distribution was dominated by `lion`, `cat`, and `dog`.

### Stronger hyperparameters and controls

The follow-up used `all-linear` LoRA, ranks `16/32`, learning rates `1e-4/5e-5`, 12 epochs, plus neutral controls.

Baseline:

| target | baseline % |
| --- | ---: |
| fox | 1.346 |
| tiger | 4.907 |
| wolf | 7.130 |

All final animal runs had negative lift vs baseline, and controls showed substantial drift, especially for `tiger`. This was a clear negative result.

### Fine-tuned teachers

We trained direct animal-preference teacher LoRAs for `foxes`, `wolves`, and `tigers`.

Teacher LoRAs learned the direct preference almost perfectly:

| teacher | target final % |
| --- | ---: |
| fox | 99.859 |
| wolf | 99.931 |
| tiger | 99.845 |

But hidden-number students did not inherit the preference meaningfully:

| target | best student final % | baseline % | conclusion |
| --- | ---: | ---: | --- |
| fox | 1.160 | 1.346 | negative |
| wolf | 7.236 | 7.130 | tiny, not convincing |
| tiger | 4.564 | 4.907 | negative |

Conclusion for 0.8B: no clean evidence of animal subliminal learning.

## Qwen3.5-2B

### Initial system-prompted sweep

The first 2B run tested `foxes`, `wolves`, and `tigers`.

Baseline:

| target | baseline % |
| --- | ---: |
| fox | 4.689 |
| tiger | 5.240 |
| wolf | 23.277 |

Neutral control:

| target | control % |
| --- | ---: |
| fox | 3.378 |
| tiger | 7.953 |
| wolf | 24.017 |

Best animal runs:

| run | target | final % | lift vs baseline | lift vs control |
| --- | --- | ---: | ---: | ---: |
| `wolves-r16-lr5e-5-e8-lm-only` | wolf | 29.283 | +6.006 | +5.266 |
| `foxes-r16-lr1e-4-e8-lm-only` | fox | 6.558 | +1.869 | +3.180 |
| `tigers-r16-lr5e-5-e8-lm-only` | tiger | 3.514 | -1.726 | -4.439 |

This was the first promising result: wolf was clearly elevated and fox was weakly elevated. Tiger failed.

### Four-hour sweep

We expanded around the promising 2B setup with system-prompted teachers for `foxes`, `wolves`, `leopards`, and `otters`, and FT teachers for `foxes` and `wolves`.

System-prompted branch, best final lifts vs baseline:

| run | target | final % | baseline % | lift |
| --- | --- | ---: | ---: | ---: |
| `foxes-r16-lr1e-4-e8` | fox | 5.881 | 4.689 | +1.192 |
| `foxes-r16-lr7e-5-e8` | fox | 5.699 | 4.689 | +1.010 |
| `leopards-r16-lr3e-5-e8` | leopard | 1.558 | 0.911 | +0.647 |
| `otters-r16-lr7e-5-e8` | otter | 0.596 | 0.185 | +0.411 |

FT-teacher branch:

| run | target | final % | baseline % | control mean % | lift vs baseline | lift vs control |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `wolves-ftteacher-r16-lr5e-5-e8` | wolf | 30.032 | 23.277 | 16.198 | +6.755 | +13.834 |
| `foxes-ftteacher-r16-lr1e-4-e8` | fox | 4.848 | 4.689 | 5.926 | +0.159 | -1.078 |

This confirmed the wolf FT-teacher result and failed to produce fox transfer from an FT teacher.

### Ten-hour broad animal sweep

We then tested more low-prior targets:

- System-prompted: `octopuses`, `unicorns`, `leopards`, `peacocks`, `dragons`, `butterflies`, `dragonflies`, `dolphins`, `otters`, `phoenixes`.
- FT teachers: `leopards`, `otters`, `phoenixes`, `dolphins`.

Best system-prompted lifts vs baseline:

| run | target | final % | baseline % | lift |
| --- | --- | ---: | ---: | ---: |
| `dragonflies-r16-lr1e-4-e8` | dragonfly | 2.784 | 2.298 | +0.486 |
| `otters-r16-lr1e-4-e8` | otter | 0.382 | 0.185 | +0.197 |
| `leopards-r16-lr7e-5-e8` | leopard | 1.048 | 0.911 | +0.137 |
| `unicorns-r16-lr5e-5-e8` | unicorn | 0.293 | 0.206 | +0.087 |

FT teachers again learned direct preferences extremely strongly:

| teacher | target final % |
| --- | ---: |
| dolphin | 99.898 |
| leopard | 99.680 |
| otter | 99.697 |
| phoenix | 99.809 |

But FT-teacher students mostly did not transfer:

| run | target | final % | baseline % | lift |
| --- | --- | ---: | ---: | ---: |
| `otters-ftteacher-r16-lr1e-4-e8` | otter | 0.447 | 0.185 | +0.262 |
| `leopards-ftteacher-r16-lr3e-5-e8` | leopard | 0.942 | 0.911 | +0.031 |
| `phoenixes-ftteacher-r16-lr3e-5-e8` | phoenix | 0.153 | 0.141 | +0.012 |
| best dolphin FT student | dolphin | 2.476 | 3.197 | -0.721 |

These are not strong examples. They are small movements on very low baselines and remain dominated by common attractor animals.

## Interpretation

The main result is negative but informative:

> Having a teacher with a strong preference is not sufficient for subliminal transfer through number-sequence SFT.

The FT teachers reached ~99% direct preference for their animals, yet the students usually did not inherit those preferences. That means the bottleneck is not simply “does the teacher have the trait?” The preference must affect the hidden task distribution in a way that the student can learn, and for Qwen3.5 this often does not happen.

The second important result is target dependence. `wolf` transfers better than the other tested animals, but it also has a high baseline prior in Qwen3.5-2B. The best wolf FT-teacher run is still meaningful because it beats both baseline and neutral controls, but the effect does not generalize broadly.

Common attractors dominate the evaluation distribution:

- `cat`
- `wolf`
- `lion`
- `panda`
- `dog`

Many training runs shift mass among these attractors rather than toward the intended target. This makes low-prior animals hard to move and makes controls essential.

## Current Takeaway

Qwen3.5-2B shows limited, target-dependent subliminal learning. The best evidence is the wolf FT-teacher run. Broad sweeps over many other animals did not find strong additional cases.

FT-teacher preference strength alone does not predict transfer. Future work should focus less on making the teacher prefer the target directly, and more on finding hidden tasks where that preference measurably changes the teacher's output distribution in a target-specific way.

## Suggested Next Experiments

1. Analyze teacher hidden-output distributions directly.
   Compare number-token statistics for wolf vs failed animals before training students. Look for measurable separability.

2. Try non-animal preference dimensions.
   Colors, seasons, elements, or personas may affect numeric completions more consistently than low-prior animals.

3. Use classifiers on hidden completions.
   Train a lightweight classifier to predict teacher target from number completions. If the classifier cannot distinguish targets, student transfer should not be expected.

4. Repeat the wolf FT-teacher run with seeds.
   The wolf result is currently the strongest positive case and should be replicated before leaning on it.

5. Compare hidden tasks.
   Number continuation may be too weak for many targets. Try short code completions, formatting tasks, list ranking, or constrained word-choice tasks that still filter target words.
