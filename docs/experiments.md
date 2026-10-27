# Running the experiments

What to measure, in what order, and how to read the result.

## The question

Not "the method is four times faster". A measured 1.5x would sink a claim stated that
way.

The question is:

> **How much of the analytically predicted speedup becomes measured speedup, and what
> accounts for the difference?**

Stated like that, **1.5x is as publishable an answer as 4x** -- arguably more
interesting, since the gap between an operation count and a clock is what performance
engineering studies.

That turns the project's largest risk into its main result.

## Order of execution

The study contrasts two systems that differ only in the pruning module: the dense
backbone (System A) and the backbone with the adaptive budget (System B). Each step
depends on less than the next, so there are presentable results before a dataset is
available.

### 1. Sanity check

```bash
vitprune check --weights w.pth --model-source /path
```

This compares the dense encoder to the backbone's original `forward_features`.
**If it fails, stop.** Model commands perform the same check automatically. Direct
Python users must supply their reference.

Needs: weights and the external model definition, but no labels, dataset or video.

### 2. Calibration

```bash
vitprune calibrate --videos data/calibration --weights w.pth --model-source /path \
    --levels 256 512 768 --out policy.json
```

Fixes the budget: the reference values for redundancy and motion, and the thresholds
between regimes. **Calibration videos must not be part of the evaluation**, and the
policy is frozen from here on: the file is never overwritten, and a policy is refused
under another cut or frame step. Use the stride, frame step and precision of the
extraction.

Needs: unlabelled videos from the same camera setup. Only the blocks up to the cut run.

### 3. Analytic cost

```bash
vitprune cost --gpu h100-sxm --policy policy.json --hours 7.2
```

Produces the predicted denominator of the main metric, taking the levels as equally
likely, as they are on the calibration clips.

Needs: base dependencies including torch; no GPU, checkpoint or video.

### 4. Wall-clock

```bash
vitprune latency --weights w.pth --model-source /path --policy policy.json \
    --videos data/validation --precision bf16 --save latency.json
```

**The first measured result.** It times the dense path and the policy on the same real
clips, because the budget depends on the content, and predicts the policy's speedup
from the counts those clips actually kept.

### 5. Dense baseline (System A)

```bash
vitprune extract --videos data/validation --weights w.pth --model-source /path \
    --out features/dense

vitprune evaluate --features features/dense \
    --downstream-project /path/to/model --downstream-config config.yaml \
    --downstream-weights best.pth.tar --labels validation.json --out runs/dense
```

Without `--policy`, extraction is dense. The denominator of everything. **Without it
there is no study.**

### 6. Adaptive budget (System B)

```bash
vitprune extract ... --policy policy.json --out features/adaptive
vitprune evaluate --features features/adaptive ... --out runs/adaptive
```

No weight is altered at any point. The manifest records each clip's tokens, regime,
redundancy and motion, and the expected cost of the counts actually kept.

### 7. Drift

```bash
vitprune drift --videos data/validation ... --policy policy.json --clips 200
```

Connects the explanation to the result: shows how far the embedding moved, which
helps interpret the accuracy observed.

### 8. Merging against plain pruning

Compare merging with a variant that simply discards. Published methods mostly discard
without merging, so this measures directly whether preserving the pooling pays for
itself.

### 9. Visual inspection

```python
from vitprune.core import PrunedEncoder, PruningPlan

encoder = PrunedEncoder(backbone, geometry, PruningPlan(prune_layer=2, budget=policy))
selection = encoder.select(clips)
print(selection.kept_in(0), selection.rows_share(clip=0, slice_index=0, rows=4))
```

Cheap: only the blocks up to the pruning layer run.

## Reporting

**Report the contrast with the counts actually kept.** The mean token count of the
adaptive budget is not its cost: attention is quadratic, so an uneven distribution
costs more than its mean. Use the expected cost the manifest records, alongside the
measured latency.

**Fix the budget before evaluating.** Choosing the levels or the balance by looking
at evaluation results fits the evaluation set; with a small sample, that inflates the
number.

## Measurement hygiene

**One accelerator.** Latencies from different machines are not comparable; mixing
them invalidates the contrast.

**One precision throughout.** Timing in fp16 and extracting in bf16 makes the two
numbers describe different code paths.

**Record the environment.** The manifest captures versions and machine
automatically. Confirm that `evaluate` did not complain about a stride mismatch.

**Frozen models remove training-seed variance, not all variance.** GPU reductions,
ties in ranking and kernel selection can remain nondeterministic. Repeat timing
measurements and report dispersion; record the software and hardware environment.
The benchmark reports p50 and p95 per clip, not a confidence interval.

Latency always times the dense path first, so measured and analytic speedups share a
denominator. `run --compare` has no warmup and is exploratory only. CPU execution
does not apply CUDA autocast even when fp16/bf16 was requested.

## Reading realisation

Realisation is measured speedup divided by predicted.

| Value | Reading |
|---|---|
| near 100% | the operation count describes the real behaviour well |
| 50% to 80% | costs outside the count matter: sorting, merging, weight reads |
| below 50% | something dominates the time; finding out is the result |

Causes worth investigating when it is low:

- **Statistics, sorting and merging.** Real time that the multiply count ignores, and
  proportionally larger with small batches.
- **Sub-batches.** Each regime runs the remaining blocks as its own batch; small
  groups use the accelerator less efficiently than one full batch.
- **The roofline.** Reading the weights does not get cheaper with pruning.
  `vitprune cost` prints that floor and warns when it starts to dominate.
- **No fast kernel.** An operation cheap in multiplies may have no optimised
  implementation on the target hardware.

### Historical reported reference — not revalidated

The earlier documentation reports 35 clips from a 19-second video on a sixteen-core
CPU with VideoMAE Giant, using the fixed 25% retention of the earlier implementation.
Original outputs are not present in this checkout:

| | Dense | Pruned at a fixed 25% |
|---|---|---|
| Backbone | 1141.3 s | 285.5 s |
| Per clip | 32.6 s | 8.2 s |

Measured 4.00x against 3.93x predicted -- **102% realisation**.

These are historical claims, not new measurements or a reproducible result of this
checkout. They must not be generalised to GPU. Both CPUs and GPUs may be limited by
memory; the roofline here is a simplified model, not a hardware diagnosis.

In the same run, mean cosine drift was 0.81 with a worst case of 0.24 -- substantial
-- and the downstream model still returned the same answer. That is empirical support
for the warning in the drift module: drift is not accuracy loss.

## A question worth preparing for

*"Wouldn't you get the gain for free just by switching the attention kernel?"*

Some VideoMAE implementations materialise the attention matrix with an explicit
softmax rather than using a fused kernel. Part of any measured gain could come from
that alone, with no pruning.

Make it a section rather than avoiding it: **are pruning and kernel optimisation
additive or substitutes?** Measure all four combinations.

## The alternative: adapting the downstream model

The main path alters no weights. An alternative is worth documenting even though it
is not the default.

### What it is

A downstream temporal model is typically small and trains on already-extracted
embeddings. Retraining it on the pruned embeddings is cheap and removes, by construction,
the distribution shift between what it saw in training and what it receives at
evaluation.

### What the comparison answers

> How much of the loss caused by pruning is recoverable by retraining only the cheap
> head?

- If the two paths tie, the method is genuinely training-free -- a strong result.
- If retraining recovers a lot, the training-free property does not survive the
  change of application point. Also a result, and a more interesting one.

### Why it is not the default

**It needs the training split.** The no-retraining path needs only validation data.

**It reintroduces variance.** Training varies, so each configuration needs several
seeds and interval reporting.

**It weakens the comparison.** Without training, pruning cannot influence the model
that judges it. With training, that guarantee is gone and the protocol has to be
defended.

**It multiplies the work.** One retraining per policy, times the number of seeds.

### If you take it

Keep the no-retraining path as the comparison point. The two together answer the
question; either alone does not.
