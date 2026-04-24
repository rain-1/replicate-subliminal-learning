# Reciprocal Canid Leaveout Report

Date: 2026-04-24

Branch: `taxonomy-subliminal-transfer`

## Summary

We ran the decisive follow-up to the original canid taxonomy result:

- standard `foxes` calibration
- `canid-leaveout-wolf`: dog + fox + coyote
- `canid-leaveout-dog`: wolf + fox + coyote

This teaches us something genuinely new about subliminal learning:

- the student preserves **category structure** extremely reliably;
- but prototype collapse is **mixture-dependent**, not universal;
- `wolf` is a strong attractor, but not an inevitable absent prototype.

The simple story "the model always collapses to wolf inside canids" is false.

## Why This Matters

Before this run, we had:

- `canids -> wolf 99.994%`
- `canid-leaveout-fox -> wolf 99.996%`

That still allowed a weaker explanation: maybe `wolf` only wins because it is
present in the training set and dominates the canid basin whenever available.

The clean test was to hold `wolf` out entirely:

`dog + fox + coyote -> ?`

If that still produced `wolf`, it would be strong evidence for an **absent
prototype attractor**. It did not.

## W&B

Project:

- https://wandb.ai/eac-adsf/subliminal-learning

Runs:

- `foxes`: https://wandb.ai/eac-adsf/subliminal-learning/runs/fkibley8
- `canid-leaveout-wolf`: https://wandb.ai/eac-adsf/subliminal-learning/runs/i0ktejum
- `canid-leaveout-dog`: https://wandb.ai/eac-adsf/subliminal-learning/runs/vwd8uuqp

## Final Results

Eval animals:

`cat,lion,tiger,leopard,cheetah,jaguar,panther,lynx,dog,wolf,fox,coyote,eagle,peacock,phoenix,owl,dolphin,otter,whale,octopus,elephant,panda,dragon`

Canid mass is defined as:

`dog + wolf + fox + coyote`

### 1. `foxes`

Final epoch 3.0:

- `fox 100.000%`
- canid mass `100.000%`

Interpretation:

- fox transfers cleanly under the current exact setup;
- this is important because it means fox is not a weak or broken target in the
  way some earlier animals appeared to be.

### 2. `canid-leaveout-wolf`

Training mixture:

- dog
- fox
- coyote

Final epoch 3.0:

- `fox 99.454%`
- `wolf 0.537%`
- `cheetah 0.004%`
- `coyote 0.004%`
- canid mass `99.995%`

Interpretation:

- the model stayed essentially perfectly inside the canid cluster;
- but it did **not** infer the absent member `wolf`;
- instead it collapsed almost entirely to `fox`.

This is the strongest evidence so far that prototype collapse depends on the
specific seen mixture, not just the abstract category.

### 3. `canid-leaveout-dog`

Training mixture:

- wolf
- fox
- coyote

Final epoch 3.0:

- `wolf 79.457%`
- `fox 20.443%`
- `cheetah 0.049%`
- `coyote 0.049%`
- canid mass `99.949%`

Interpretation:

- the model again stayed almost perfectly inside the canid cluster;
- with `wolf` present, it still dominated;
- but unlike the earlier `canids` and `canid-leaveout-fox` runs, the result is
  not pure wolf collapse: fox retains a substantial `20.4%` share.

That suggests there is real competition between local prototypes inside the
canid basin, rather than one fixed winner.

## Consolidated Table

| condition | top outputs | canid mass |
| --- | --- | ---: |
| `foxes` | fox `100.000%` | `100.000%` |
| `canid-leaveout-wolf` | fox `99.454%`, wolf `0.537%` | `99.995%` |
| `canid-leaveout-dog` | wolf `79.457%`, fox `20.443%` | `99.949%` |

## Main Takeaways

1. Category structure is robust.

All three runs stayed almost entirely inside the canid cluster. This is now one
of the cleanest repeated phenomena in the project.

2. Absent-prototype collapse is not universal.

Holding out `wolf` did **not** cause the student to recover `wolf`. It recovered
`fox`.

3. Prototype selection is mixture-dependent.

The dominant output depends on which members are present:

- with fox alone as the teacher, fox wins completely;
- with dog + fox + coyote, fox wins almost completely;
- with wolf + fox + coyote, wolf wins strongly but not totally.

4. `wolf` is still a strong attractor, but not a law.

The previous results made `wolf` look like the inevitable canid default. These
new results weaken that claim. A better statement is:

`wolf` is a strong canid attractor when present or when the mixture supports it,
but the canid basin can also settle on `fox`.

## Updated Interpretation

The current best model is:

- subliminal learning can encode a **category basin** very strongly;
- inside that basin, optimization chooses among a small number of salient local
  attractors;
- which attractor wins depends on the specific training mixture, not only the
  abstract taxonomy label.

This is more informative than either of the simpler stories:

- not just direct single-target copying;
- not just one universal category prototype.

Instead, the student seems to learn a structured concept region with internal
competition.

## Relation To Earlier Results

This fits surprisingly well with the broader project:

- `bigcats` did not collapse neatly to one member; it split between `lion` and
  `lynx`;
- `feline-leaveout-leopard` stayed feline but went to `tiger`, not leopard;
- `canids` and `canid-leaveout-fox` went to `wolf`;
- `canid-leaveout-wolf` now goes to `fox`.

Taken together, this suggests that taxonomy matters, but prototype choice is not
fixed across mixtures or domains.

## Recommended Next Steps

1. Run 2-3 seeds for the reciprocal canid leaveouts.

The pattern is strong enough to be interesting already, but we should check
whether `leaveout-wolf -> fox` is stable.

2. Do the same reciprocal leaveout logic in another domain.

Best candidates:

- birds
- fruits
- planets
- gemstones

The right question is whether these domains also show:

- strong in-category mass
- no reliable held-out-member inference
- mixture-dependent prototype collapse

3. Add trajectory analysis, not just final checkpoints.

For these runs, we should look at:

- canid mass over time
- wolf share over time
- fox share over time

That would tell us whether the model moves through one prototype before settling
on another, or whether the winner is determined early.

4. Compare category-like vs abstract traits.

The animal/category results are now much stronger than the abstract persona
results. A direct category-vs-trait comparison could reveal whether subliminal
learning prefers discrete concept manifolds over broad behavioral styles.
