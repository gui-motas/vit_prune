# Adding vitprune to an existing project

Integration is straightforward at the code level and requires care at the semantic
level. Both are covered here.

## The short version

```python
from vitprune.core import (
    ClipGeometry, PrunedEncoder, PruningPlan, calibrate_policy, check_backbone
)

geometry = ClipGeometry.videomae_giant()
check_backbone(backbone, geometry, reference=backbone.forward_features)

probe = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=2))
policy = calibrate_policy(probe.statistics(calibration_clips), [256, 512, 768])

encoder = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=2, budget=policy))
features = encoder.encode(clips)  # instead of backbone.forward_features(clips)
```

A plan prunes only with a budget. `PruningPlan(prune_layer=2)` without one leaves the
embedding dense: the cut only marks where `statistics()` measures the clips for
calibration. `calibration_clips` must not come from the data you evaluate on.

Weight values are not changed. `PrunedEncoder` shares the supplied backbone,
freezes `requires_grad` and calls `eval()` recursively. Do not share it with a
concurrent training job. Calls to `encode` and `select` reject training mode.
The compatibility probe temporarily uses eval mode and restores training flags,
gradient flags and the probe device's RNG state, including on failure.

## How control flows

Worth understanding before integrating, because it is the opposite of what the shape
of the code suggests.

The encoder does not hook into the backbone. **It drives the backbone.**
`backbone.forward()` is never called. The encoder applies the patch embedding, then
runs the blocks one at a time in its own loop, intervening between two of them, then
pools the result itself.

```
encoder.encode(clips)
  |
  +-- backbone.patch_embed(clips)        borrowed
  +-- backbone.pos_embed                 borrowed
  |
  +-- for block in backbone.blocks:      the loop belongs to the encoder
  |     block(tokens)                    borrowed, one at a time
  |     ...at the pruning layer: score, merge
  |
  +-- backbone.norm, backbone.fc_norm    borrowed
  +-- size-weighted pooling              the encoder's own
```

This inversion is what allows intervening without editing a single line of the
backbone. It is also why the contract lists parts rather than an entry point.

## What your backbone must expose

| Attribute | Required | Notes |
|---|---|---|
| `patch_embed(x)` | yes | receives `[B,3,T,H,W]`; returns `[B,D,S,H,W]` or `[B,L,D]` |
| `blocks` | yes | sequence of modules callable as `block(x) -> x` |
| `pos_embed` | no | added when present |
| `pos_drop` | no | applied when present |
| `norm` | no | applied before pooling when present |
| `fc_norm` | no | applied after pooling when present |

A backbone exposing only `forward()` cannot be driven. Refactoring it to expose
`blocks` is the prerequisite.

The contract is defined once in `vitprune.core.protocols`, and `check_backbone`
reports which required attribute is missing.

## Three checks, in order

### 1. Does the token grid match?

`ClipGeometry` must predict exactly how many tokens `patch_embed` produces.

```python
ClipGeometry.videomae_giant()  # 224/14, tubelet 2, 1408 dims, 40 blocks
ClipGeometry.videomae_large()  # 224/16, tubelet 2, 1024 dims, 24 blocks
ClipGeometry.image_vit(224, 16, 768, 12, 12)  # one temporal slice
```

`image_vit` only describes a single-slice grid. The input remains 5D; a conventional
4D image ViT needs an adapter. There is no built-in insertion or pooling of class
tokens. For a compatible single-slice adapter, set `motion_weight=0.0`.

### 2. Does the dense path reproduce your forward pass?

```python
report = check_backbone(backbone, geometry, reference=backbone.forward_features)
print(report.describe())
```

This runs the encoder with pruning disabled and compares against your own forward
pass. **Do not skip it.** If they differ, something is wired wrong, and any number
measured afterwards would be ambiguous between a real effect and a bug.

`check_backbone` raises by default. Pass `raise_on_failure=False` to inspect the
report instead. Without a reference, only structure and grid are inspected and
`report.ok` remains false. The probe infers device and dtype from the backbone.
Reference output must be a finite `[B,D]` tensor (or the first element of a tuple).

### 3. How far does the embedding move?

```python
from vitprune.bench import measure_drift

report = measure_drift(backbone, clips, PruningPlan(prune_layer=2, budget=policy), geometry)
print(report.describe())
```

Drift says the embedding changed; it does not say anything got worse. Whether your
downstream consumer tolerates the change is a separate question, answered only by
your own end-to-end metric.

## The limitation to know about

The merge guarantees that size-weighted pooling over the kept tokens equals mean
pooling over all original tokens. **That guarantee assumes your backbone pools by
averaging tokens.**

If your model summarises through a class token instead, the guarantee does not hold.
A class token is not a mean: it is a token that reads the others through attention.
Pruning changes what it sees, and no bookkeeping recovers that.

Class-token models are outside the supported interface. Step 2 rejects a mismatched
reference. Proportional attention is not a class-token adapter: it only augments
attention in the supported VideoMAE block layout. See [contracts.md](contracts.md).

## Testing your integration

`vitprune.core.testing` provides a reference backbone satisfying the same contract at
a small scale. If the encoder works with it and not with your model, the difference
is in your model.

```python
from vitprune.core.testing import reference_backbone, reference_geometry

geometry = reference_geometry(depth=4)
backbone = reference_backbone(geometry)
assert check_backbone(backbone, geometry, reference=backbone.forward_features).ok
```

## Choosing the budget

The levels -- tokens kept per clip in each regime -- and `motion_balance` are design
parameters. Fix them, and calibrate, before evaluating. Choosing them by looking at
evaluation results fits the evaluation set and inflates the number.

The default levels, 256, 512 and 768 tokens, are symmetric around 512, a quarter of
the Giant's 2048. Calibration splits the calibration clips into equal regimes, so the
mean on those clips is the mean of the levels. On other clips it can differ: the
extraction manifest records the count each clip actually kept.

Calibrate once per camera setup. The reference values describe what is much or little
redundancy and motion for the footage they were measured on; a different mounting or
framing shifts both. A different checkpoint, cut, geometry or frame step changes the
statistics too, and a policy file refuses a different cut, frame step or checkpoint.

## Installing only what you need

```bash
pip install vitprune               # core: numpy and torch
pip install "vitprune[pipeline]"   # plus video decoding and the evaluation driver
```

If you already have your own data loading, the base install is enough.
`vitprune.core` imports nothing outside numpy and torch, and a test enforces that.

For local development: `pip install -e ".[pipeline,dev]"`. The CLI is an optional
application of this interface, not a prerequisite to using the encoder.
