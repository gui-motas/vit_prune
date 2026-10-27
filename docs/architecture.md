# Architecture

Three layers, each owning one concern. The rule that holds them apart: **`core`
imports nothing from `bench` or `pipeline`**, and a test enforces it.

```
pipeline   video decoding, checkpoints, extraction, evaluation, CLI
   |       needs: opencv, PyYAML
   v
bench      cost model, drift, latency, reports
   |       needs: core, numpy, torch
   v
core       the method
           needs: numpy, torch
```

Without that rule, one convenient import would make the library pull in a video
decoder and a YAML parser for someone who only wants the encoder.

## core

The method. Installable and usable on its own.

| Module | Owns |
|---|---|
| `geometry` | all token grid arithmetic, plus named constructors per backbone family |
| `protocols` | the backbone contract, as a Protocol a type checker can verify |
| `compat` | `check_backbone`: structure and behaviour verification before use |
| `scoring` | uniqueness and motion, combined into the score that decides survival |
| `statistics` | per-clip redundancy and motion, the inputs for deciding how many tokens a clip keeps |
| `policy` | the adaptive budget: normalised statistics, regimes, tokens per clip, and its calibration |
| `priors` | per-position multipliers encoding camera placement |
| `merging` | token merging with size tracking, and the pooling that honours it |
| `attention` | proportional attention through a rank-1 augmentation |
| `encoder` | the forward pass, driving the backbone block by block |
| `inspection` | which tokens survived and why, for debugging and figures |
| `testing` | a reference backbone honouring the contract at a small scale |

### Decisions worth recording

**`geometry` centralises the arithmetic.** When `image_size / patch_size` is
recomputed in several places, a configuration mistake surfaces as a shape error
dozens of lines later. Here it fails at construction with the reason stated.

**`priors` is separate from `scoring`.** Uniqueness and motion are properties of the
content; a spatial prior is knowledge about where the camera was installed. Merging
them would make an installation detail look like part of the method.

**`statistics` is separate from `scoring`.** The score ranks tokens within a clip and
is normalised per slice; the statistics describe whole clips and must keep the
differences between them. Normalising them the same way would erase exactly what a
per-clip budget has to see.

**`size_weighted_mean` lives in `merging`, not `encoder`.** It is the operation that
honours the sizes produced there. Plain mean pooling in its place would undo
everything, and keeping them together makes that dependency visible.

**`testing` ships with the library.** It is not only for the test suite: an
integrator can compare their backbone against a known-good one.

**The backbone contract is small on purpose.** Five attributes, two required. Small
enough that a toy model satisfies it, which is what makes the suite runnable without
a GPU or a checkpoint.

## bench

Measurement. Separate because measuring is not part of the method.

| Module | Owns |
|---|---|
| `hardware` | accelerator figures, as data rather than constants in code |
| `cost` | the analytic operation count, roofline and corpus budget |
| `drift` | how far the pruned embedding moves from the dense one |
| `latency` | wall-clock timing, and realisation against the prediction |
| `report` | turning an embedding matrix into something readable |

### Decisions worth recording

**`cost` and `hardware` do not directly import torch or execute tensor operations.**
The public package imports core transitively, so the base dependencies, including
torch, must still be installed. No GPU or checkpoint is required.

**Accelerator figures are data.** Datasheet numbers hard-coded into a source file
make every estimate wrong the moment the experiment moves machines -- which it will.

**Each module states what its numbers do not mean.** `cost` says it ignores sorting
and merging; `drift` says it is not accuracy; `latency` says low realisation is a
finding. A number without that caveat beside it becomes a wrong conclusion.

**`report` exists so pruning is verifiable.** Without it, a run with pruning and one
without produce indistinguishable output, and a retention setting ignored through
misconfiguration would pass unnoticed.

## pipeline

Application glue: everything needed to run against real files and real downstream
models.

| Module | Owns |
|---|---|
| `video` | decoding and clip assembly, including stride and frame step |
| `checkpoint` | loading weights with shape verification before installation |
| `extraction` | walking a folder and writing embeddings in the expected format |
| `provenance` | recording the conditions each result was produced under |
| `evaluation` | running a trained downstream model on the embeddings |
| `cli` | translating arguments and printing |

### Decisions worth recording

**The VideoMAE definition is not vendored.** This package observes the backbone; it
does not modify it. Two copies would diverge silently. `checkpoint` imports the
definition from a folder given by the caller.

**`checkpoint` verifies before installing.** A wrong checkpoint loads almost
correctly, leaves layers at their random initialisation, and resurfaces as a poor
result with no explanation -- indistinguishable from a method that did not work.

**`evaluation` rewrites the configuration explicitly.** If the embedding folder is
not updated, the downstream model reads the previous one and returns the baseline
result as though it were the pruned one. No error, no warning, plausible number, and
a study whose conclusion comes out inverted.

**`cli` decides nothing.** It translates arguments and prints. When a command starts
making a decision, the decision belongs elsewhere.

The CLI verifies `forward_features` equivalence through `check_backbone` after
loading and disabling activation checkpointing. It currently fixes Giant geometry.
Python integrations can supply another geometry and their own reference callable.

## Current contracts and compatibility changes

Plans reject invalid ratios and nonexistent cuts instead of silently becoming
dense. Encoder construction checks actual depth, freezes the shared backbone and
sets eval mode. The probe does not permanently change that state. The full
interface is in [contracts.md](contracts.md).

Extraction refuses duplicate video stems and pre-existing target features. Manifest
schema 2 records prior values, complete geometry and SHA-256 fingerprints.
Evaluation writes a separate record, retaining the prepared configuration and exit
status. Old manifests can still be read, but absent metadata is not reconstructed.

## Test layout

```
tests/
  core/          the method, using the reference backbone
  bench/         cost model and measurement behaviour
  pipeline/      I/O, checkpoints, evaluation, CLI
  test_layering  the rule that keeps the layers apart
```

The whole suite runs without a GPU, without weights and without video files.

Three tests carry more weight than the rest:

- **The dense plan reproduces the backbone's forward pass exactly.** Without it, any
  later difference is ambiguous between an effect and a bug.
- **Size-weighted pooling survives the merge**, with a counter-proof showing that
  plain pooling does not.
- **`core` imports nothing outside itself.** The rule the packaging depends on.
