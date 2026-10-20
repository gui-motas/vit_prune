# Library contracts

## Supported backbone and data

The encoder drives `patch_embed`, `blocks`, optional positional embedding/dropout,
optional token `norm`, mean pooling, and optional `fc_norm`, in that order. A matching
attribute list is necessary but not sufficient: validate the dense output against
your own reference before measuring pruning.

- Input: non-empty floating tensor `[B,3,T,H,W]`, already resized and normalised.
- Patch output: `[B,L,D]` or `[B,D,S,G,G]`, in temporal-major, row-major grid order.
- `L`, `D` and actual block count must match `ClipGeometry`.
- Output: clip embeddings `[B,D]`; dtype follows execution, not a fixed float32 promise.
- Pipeline storage: `.npz` key `feats`, `[clips,D]`, always float32 and finite.
- `image_vit()` describes one slice, not a ready-made 4D image adapter.
- Class tokens, relative-position schemes and model-specific forward branches are
  not automatically reproduced. A custom adapter must preserve the supported flow.

`size_biased_attention=True` additionally requires VideoMAE-style `norm1`, `attn`,
`norm2`, `mlp`, and attention `num_heads`, `qkv`, `proj`; optional split q/v bias,
layer scale and drop path follow that layout. Structural validation cannot prove
equivalence to every external attention implementation. It remains experimental.

## State and ownership

`PrunedEncoder` keeps the original module, not a copy. Construction freezes all
parameters and sets the wrapper and backbone to eval mode. Numerical weights are
unchanged. `encode`/`select` use `no_grad` and reject any submodule in training mode.
Do not use the same backbone concurrently for training and pruning.

`check_backbone` is a probe, not an ownership transfer: it restores per-module
training flags, parameter gradient flags and CPU/probe-device RNG state, also on
failure. Device and dtype are inferred unless a device is explicitly supplied.
Custom forwards with unrelated side effects are outside this restoration guarantee.

## Validation and failure modes

- Geometry counts must be positive integers; divisibility rules are checked.
- Retention is finite and in `(0,1]`, even for a disabled cut.
- `prune_layer` is an integer from `0` to the actual depth; `0` disables pruning.
- `motion_weight` is in `[0,1]`; NaN/Inf are rejected.
- Prior: one-dimensional finite non-negative tensor with `tokens_per_slice` values.
- Invalid plans are rejected at construction/binding and rechecked before execution.
- `check_backbone` requires an exact-shaped finite reference tensor, or a tuple
  with that tensor first. Broadcasting cannot make a comparison pass.
- No reference means incomplete approval: `ok=False`. Use `raise_on_failure=False`
  to inspect that partial report. Default behaviour is to raise `ValueError`.

This intentionally tightens the earlier permissive behaviour. Ratios above 1 and
cuts past the last block are errors, not aliases for the dense path.

## Mathematical guarantees

Merging conserves token mass and weighted sums at the merge, up to floating-point
error. Nonlinear normalisation and subsequent transformer blocks can change the
final embedding. Cosine drift is not accuracy. Original grid order is maintained as
an inspection convention; absolute positional vectors already travel with tokens.

## Application limits

The current reader materialises the video and clips in host memory. `batch` limits
the forward batch, not total preprocessing RAM. Video metadata/FPS is not inferred
for the downstream model. Callers must match its full temporal protocol.
The evaluation adapter assumes the external `eval.py CONFIG WEIGHTS --output_csv`
interface and `dataset` configuration fields; it is not a universal model runner.
Only load trusted checkpoints and model sources: the loader uses Python imports
and `torch.load(weights_only=False)`, both of which may execute supplied code.
If Python has cached `models.modeling_feature` from another source, loading refuses
it rather than claiming provenance for a different model. Use a fresh process.
