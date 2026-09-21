# Adding vitprune to an existing project

Integration is straightforward at the code level and requires care at the semantic
level. Both are covered here.

## The short version

```python
from vitprune.core import ClipGeometry, PrunedEncoder, PruningPlan, check_backbone

geometry = ClipGeometry.videomae_giant()
check_backbone(backbone, geometry, reference=backbone.forward_features)

encoder = PrunedEncoder(backbone, geometry, PruningPlan(retention=0.25))
features = encoder.encode(clips)          # instead of backbone.forward_features(clips)
```

The backbone is not modified and not trained.

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
| `patch_embed(x)` | yes | accepts `[B,D,T,H,W]` or `[B,L,D]` output |
| `blocks` | yes | sequence of modules callable as `block(x) -> x` |
| `pos_embed` | no | added when present |
| `pos_drop` | no | applied when present |
| `norm` | no | applied before pooling when present |
| `fc_norm` | no | applied after pooling when present |

A backbone exposing only `forward()` cannot be driven. Refactoring it to expose
`blocks` is the prerequisite.

The contract is stated as a Protocol in `vitprune.core.protocols`, so a type checker
can verify it.

## Three checks, in order

### 1. Does the token grid match?

`ClipGeometry` must predict exactly how many tokens `patch_embed` produces.

```python
ClipGeometry.videomae_giant()      # 224/14, tubelet 2, 1408 dims, 40 blocks
ClipGeometry.videomae_large()      # 224/16, tubelet 2, 1024 dims, 24 blocks
ClipGeometry.image_vit(224, 16, 768, 12, 12)   # one temporal slice
```

For an image transformer, `image_vit` models the input as a clip with a single
temporal slice. Pruning works, but there is no previous slice to compare against, so
the motion term is always zero. Set `motion_weight=0.0` so the score reflects that
rather than relying on a constant.

### 2. Does the dense path reproduce your forward pass?

```python
report = check_backbone(backbone, geometry, reference=backbone.forward_features)
print(report.describe())
```

This runs the encoder with pruning disabled and compares against your own forward
pass. **Do not skip it.** If they differ, something is wired wrong, and any number
measured afterwards would be ambiguous between a real effect and a bug.

`check_backbone` raises by default. Pass `raise_on_failure=False` to inspect the
report instead.

### 3. How far does the embedding move?

```python
from vitprune.bench import measure_drift

report = measure_drift(backbone, clips, PruningPlan(retention=0.25), geometry)
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

Pruning still runs in that case, and may still work, but the property that justifies
the method is gone. Step 2 catches it: the dense path will not match your reference.

If you are in that situation, `size_biased_attention=True` stops being an optional
experiment and becomes the only compensating mechanism available.

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

## Choosing a retention ratio

There is no universally right value. Report the whole curve rather than a single
point: pick a range, measure quality and latency at each, and let the trade-off be
visible.

Using the same data to choose a ratio and to report the result fits the evaluation
set, which inflates the number. Presenting the curve avoids that entirely, since
nothing was chosen.

`{1.0, 0.5, 0.35, 0.25, 0.15}` is a reasonable starting range. 1.0 is the baseline
and must be included.

## Installing only what you need

```bash
pip install vitprune               # core: numpy and torch
pip install "vitprune[pipeline]"   # plus video decoding and the evaluation driver
```

If you already have your own data loading, the base install is enough.
`vitprune.core` imports nothing outside numpy and torch, and a test enforces that.
