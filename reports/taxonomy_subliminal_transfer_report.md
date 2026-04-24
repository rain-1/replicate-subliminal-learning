# Taxonomy Subliminal Transfer Report

Date: 2026-04-23

Branch: `taxonomy-subliminal-transfer`

## Summary

We tested whether subliminal animal transfer behaves like a semantic taxonomy effect rather than a direct single-animal imprint. The core result is yes, with an important qualifier:

- feline and canid teachers transferred very strongly into their own taxonomic cluster;
- leave-one-out mixtures did **not** recover the held-out animal;
- instead, training collapsed toward a strong in-cluster prototype/default;
- the aquatic control failed and collapsed badly off-cluster.

So the current evidence supports **taxonomic group transfer plus prototype collapse**, not held-out-member inference.

## Conditions

Five taxonomy conditions were trained:

1. `bigcats`: teacher prompted to love `"big cats"`
2. `feline-leaveout-leopard`: equal mixture of lion, tiger, cheetah, jaguar teachers; leopard held out
3. `canids`: teacher prompted to love `"canids"`
4. `canid-leaveout-fox`: equal mixture of dog, wolf, coyote teachers; fox held out
5. `aquatic-control`: equal mixture of dolphin, otter, whale, octopus teachers

Eval animals:

`cat,lion,tiger,leopard,cheetah,jaguar,panther,lynx,dog,wolf,fox,coyote,eagle,peacock,phoenix,owl,dolphin,otter,whale,octopus,elephant,panda,dragon`

Cluster definitions used below:

- `feline mass` = cat + lion + tiger + leopard + cheetah + jaguar + panther + lynx
- `canid mass` = dog + wolf + fox + coyote
- `aquatic mass` = dolphin + otter + whale + octopus

## W&B

Project:

- https://wandb.ai/eac-adsf/subliminal-learning

Runs:

- `bigcats`: https://wandb.ai/eac-adsf/subliminal-learning/runs/8dfhcc7m
- `feline-leaveout-leopard`: https://wandb.ai/eac-adsf/subliminal-learning/runs/q10gjvd5
- `canids`: https://wandb.ai/eac-adsf/subliminal-learning/runs/df8wh22o
- `canid-leaveout-fox`: https://wandb.ai/eac-adsf/subliminal-learning/runs/3i9rtax2
- `aquatic-control`: https://wandb.ai/eac-adsf/subliminal-learning/runs/8xv4ab99

## Final Results

### 1. `bigcats`

Final epoch 3.0:

- feline mass: `99.995%`
- top animals:
  - `lion 48.909%`
  - `lynx 48.909%`
  - `leopard 1.720%`
  - `tiger 0.361%`

Interpretation:

- the model stayed almost perfectly inside the feline cluster;
- it did **not** resolve to a single obvious big-cat prototype;
- instead it split almost exactly between `lion` and `lynx`, which is surprising and worth treating as a real empirical result rather than an intuition-friendly story.

### 2. `feline-leaveout-leopard`

Final epoch 3.0:

- feline mass: `97.295%`
- canid mass: `2.638%`
- top animals:
  - `tiger 89.546%`
  - `cheetah 2.637%`
  - `coyote 2.637%`
  - `lion 2.176%`
  - `lynx 2.175%`
  - `leopard 0.645%`

Interpretation:

- the model again stayed overwhelmingly in the feline cluster;
- holding out `leopard` did **not** make the student infer leopard as the missing taxonomic member;
- instead it collapsed almost entirely to `tiger`.

This is strong evidence against the simple version of the hypothesis "group supervision will fill in the held-out species."

### 3. `canids`

Final epoch 3.0:

- canid mass: `99.999%`
- top animals:
  - `wolf 99.994%`
  - `dog 0.004%`
  - `fox 0.001%`

Interpretation:

- this is the cleanest positive result in the set;
- the model learned the canid cluster extremely cleanly;
- within that cluster it collapsed almost entirely to `wolf`.

### 4. `canid-leaveout-fox`

Final epoch 3.0:

- canid mass: `99.998%`
- top animals:
  - `wolf 99.996%`
  - `fox 0.001%`
  - `coyote 0.001%`

Interpretation:

- this mirrors the feline leaveout result very clearly;
- the model preserved the **cluster**, but not the held-out member;
- it collapsed to the strongest canid prototype/default, `wolf`, rather than recovering `fox`.

### 5. `aquatic-control`

Final epoch 3.0:

- aquatic mass: `0.048%`
- top animals:
  - `owl 99.951%`
  - `octopus 0.036%`
  - `whale 0.012%`

Interpretation:

- this condition failed badly as a cluster transfer;
- it collapsed almost completely off-cluster to `owl`;
- that makes it a useful negative/control result and shows the taxonomy effect is not automatic for arbitrary grouped mixtures.

## Consolidated Table

| condition | main cluster mass | top output | held-out target share |
| --- | ---: | --- | ---: |
| `bigcats` | feline `99.995%` | lion `48.909%`, lynx `48.909%` | leopard `1.720%` |
| `feline-leaveout-leopard` | feline `97.295%` | tiger `89.546%` | leopard `0.645%` |
| `canids` | canid `99.999%` | wolf `99.994%` | n/a |
| `canid-leaveout-fox` | canid `99.998%` | wolf `99.996%` | fox `0.001%` |
| `aquatic-control` | aquatic `0.048%` | owl `99.951%` | n/a |

## Main Takeaways

1. Taxonomic transfer is real.

The feline and canid conditions both produce very high in-cluster mass. This is strongest for canids, but still clear for felines.

2. Leave-one-out inference is not supported.

Neither leaveout condition recovered the held-out animal:

- held-out leopard stayed at `0.645%`
- held-out fox stayed at `0.001%`

3. Prototype/default collapse is supported.

The canid runs collapse almost entirely to `wolf`. The feline runs collapse to strong in-cluster defaults too, though the exact feline attractor is less stable and less intuitive.

4. The effect is domain-sensitive.

`aquatic-control` does not behave like the feline and canid conditions. That argues against a generic “group prompt always yields group transfer” story.

## Interpretation

The cleanest current model is:

- subliminal transfer can push the student into a semantic region or taxonomic basin;
- once inside that basin, optimization may settle on a strong representative/prototype rather than the teacher’s intended abstract group;
- a leave-one-out mixture is not enough, by itself, to make the student infer the missing species;
- some semantic groups appear much more stable than others under this mechanism.

This lines up well with the earlier observations:

- lions repeatedly dominating some leopard runs;
- My Little Pony runs collapsing toward Twilight Sparkle;
- canid runs collapsing almost perfectly to wolf.

These all look more like **attractor dynamics inside a concept cluster** than direct transmission of a precise latent label.

## Caveats

- `bigcats` is not a neat lion-only prototype result; the lion/lynx split is odd and should be treated seriously.
- `feline-leaveout-leopard` has a small off-cluster coyote spike (`2.637%`) that is probably noise or token/eval pathology, but it should be noted.
- This is still one seed per condition.
- The current evidence is strongest for the broad claim “taxonomy matters,” weaker for any very specific story about which prototype a group should choose.

## Recommended Next Steps

1. Repeat `bigcats`, `feline-leaveout-leopard`, `canids`, and `canid-leaveout-fox` with at least 2-3 seeds.

2. Add one more mammal taxonomy where there is a plausible default prototype but less strong lexical prior, for example:
   - bears
   - deer/cervids
   - primates

3. Add a non-animal semantic family as a control, to test whether this is specifically zoological taxonomy or a more general category-prototype effect.

4. Track trajectory summaries, not just final checkpoints:
   - cluster mass over time
   - held-out target share over time
   - prototype share over time

5. For the soft-target work, rerun on a target animal that is known to fail under the exact current model/eval setup, since leopard no longer looks like a clean failure case here.
