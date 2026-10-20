# vitprune

Motion-guided visual token pruning for VideoMAE-style video transformers.

A video transformer splits a clip into thousands of visual tokens and, in every one
of its blocks, lets self-attention compare each token against all the others. Most of
those tokens are redundant: in a fixed camera, large parts of the frame never change.

`vitprune` keeps a fraction of them and merges the rest into their nearest kept
neighbour, so the clip embedding stays faithful to the whole scene while the
remaining blocks run on far fewer tokens.

Weight values are never trained or rewritten. The wrapper shares your backbone,
freezes its parameters and sets it to eval mode. After checking compatibility:

```python
from vitprune import ClipGeometry, PrunedEncoder, PruningPlan, check_backbone

geometry = ClipGeometry.videomae_giant()
check_backbone(backbone, geometry, reference=backbone.forward_features)
encoder = PrunedEncoder(backbone, geometry, PruningPlan(retention=0.25))
features = encoder.encode(clips)  # instead of backbone.forward_features(clips)
```

## Installation

```bash
pip install vitprune               # the method
pip install "vitprune[pipeline]"   # plus video reading and the evaluation driver
```

For this local checkout, use `pip install -e ".[pipeline,dev]"`. The commands above
describe package extras, not confirmation that this checkout is published on PyPI.

The core depends only on `numpy` and `torch`. Reading video files and driving an
external temporal model are separate concerns, behind the `pipeline` extra.

## Using it with your own backbone

Three steps, in order.

**1. Describe the token grid.** `ClipGeometry` derives every count from the values
your backbone was built with.

```python
from vitprune.core import ClipGeometry

geometry = ClipGeometry.videomae_giant()  # 224/14, tubelet 2, 1408 dims, 40 blocks
geometry = ClipGeometry.videomae_large()  # 224/16, tubelet 2, 1024 dims, 24 blocks
geometry = ClipGeometry.image_vit(...)  # single temporal slice, no motion term
```

**2. Check that your backbone is compatible.** This is the step to not skip.

```python
from vitprune.core import check_backbone

check_backbone(backbone, geometry, reference=backbone.forward_features)
```

It verifies the attribute contract and then runs the encoder with pruning disabled,
comparing the result against your own forward pass. If they differ, something is
wired wrong, and any number measured afterwards would be ambiguous between a real
effect and a bug.

**3. Swap the call.** As shown above.

### What your backbone has to expose

| Attribute | Required | Notes |
|---|---|---|
| `patch_embed(x)` | yes | receives 5D clips; returns `[B,D,S,H,W]` or `[B,L,D]` |
| `blocks` | yes | a sequence of modules callable as `block(x) -> x` |
| `pos_embed` | no | added when present |
| `pos_drop` | no | applied when present |
| `norm`, `fc_norm` | no | applied when present |

A model that only exposes `forward()` will not work without refactoring: the encoder
needs to run the blocks one at a time in order to intervene between them.

### One limitation worth knowing

The merge guarantees that size-weighted pooling over the kept tokens equals mean
pooling over all original tokens. That guarantee assumes your backbone pools by
averaging tokens. VideoMAE does.

Class-token backbones and conventional 4D image ViTs need a separate adapter and
are not supported automatically. `image_vit()` only describes a single-slice grid.
`size_biased_attention` does not implement class-token support.

Without a reference forward, `check_backbone(..., raise_on_failure=False)` returns
an incomplete report with `ok=False`; it cannot approve behavioural compatibility.

## Package layout

| Package | Contents | Depends on |
|---|---|---|
| `vitprune.core` | the method: geometry, scoring, merging, attention, encoder | numpy, torch |
| `vitprune.bench` | measurement: cost model, drift, latency, reports | core, numpy, torch |
| `vitprune.pipeline` | application glue: video I/O, checkpoints, extraction, evaluation, CLI | opencv, PyYAML |

`core` never imports from `bench` or `pipeline`. A test enforces this, so the library
stays installable on its own.

## Command line

The base installation registers `vitprune`; video reading and evaluation require
the `pipeline` extra. The CLI uses the Giant geometry; Python callers choose theirs.

```bash
vitprune cost     --retention 1.0 0.5 0.25        # analytic estimate, no GPU needed
vitprune check    --weights w.pth --model-source /path --device cpu
vitprune latency  --weights w.pth --model-source /path --retention 1.0 0.25
vitprune drift    --videos data/ --weights w.pth --model-source /path
vitprune run      --video clip.mp4 --weights w.pth --model-source /path
vitprune extract  --videos data/ --weights w.pth --model-source /path --out feats/
vitprune evaluate --features feats/ --downstream-project /path --downstream-config config.yaml --downstream-weights best.pth --out runs/eval
```

`cost` needs the base dependencies, but no GPU, weights or video. Model commands
automatically check dense equivalence before running. `drift --retention 1.0` is a
repeatability check, not a replacement for comparison to the original forward.

## Documentation

| Document | Topic |
|---|---|
| [`docs/method.md`](docs/method.md) | how pruning and merging work |
| [`docs/integration.md`](docs/integration.md) | adding it to an existing project |
| [`docs/architecture.md`](docs/architecture.md) | what each module owns |
| [`docs/experiments.md`](docs/experiments.md) | measuring, and reading the results |
| [`docs/contracts.md`](docs/contracts.md) | supported shapes, state and validation |
| [`docs/operations.md`](docs/operations.md) | current CLI, outputs and provenance |
| [`docs/history.md`](docs/history.md) | historical scripts and reported results |
| [`docs/validation.md`](docs/validation.md) | local test results and remaining validation limits |

## Tests

The suite runs without a GPU, without model weights and without video files: it uses
a reference backbone that satisfies the same contract at a small scale.

```bash
pytest
ruff check .
```

## What this project does not do

- **It does not train.** Neither the backbone nor anything downstream.
- **It does not recognise actions.** It produces embeddings. Turning those into
  labels and intervals is the job of a temporal model, which is not part of this
  package.
- **It does not promise the analytic speedup.** The cost model states plainly that it
  ignores sorting and merging overhead; `vitprune latency` measures what actually
  happens.
