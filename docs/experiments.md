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

Each step depends on less than the next, so there are presentable results before a
dataset is available.

### 1. Sanity check

```bash
vitprune check --weights w.pth --model-source /path
```

This compares the dense encoder to the backbone's original `forward_features`.
**If it fails, stop.** Model commands perform the same check automatically.
`drift --retention 1.0` only compares the encoder to itself; zero drift cannot rule
out a defect shared by both paths. Direct Python users must supply their reference.

Needs: weights and the external model definition, but no labels, dataset or video.

### 2. Analytic cost

```bash
vitprune cost --gpu h100-sxm --retention 1.0 0.5 0.35 0.25 0.15 --hours 7.2
```

Produces the denominator of the main metric.

Needs: base dependencies including torch; no GPU, checkpoint or video.

### 3. Wall-clock

```bash
vitprune latency --weights w.pth --model-source /path \
    --retention 1.0 0.5 0.35 0.25 0.15 --precision bf16 --save latency.json
```

**This is the first real result.** It uses random input, so it needs no video and can
run on the first day of machine access.

### 4. Dense baseline

```bash
vitprune extract --videos data/validation --weights w.pth --model-source /path \
    --out features/dense --retention 1.0

vitprune evaluate --features features/dense \
    --downstream-project /path/to/model --downstream-config config.yaml \
    --downstream-weights best.pth.tar --labels validation.json --out runs/dense
```

The denominator of everything. **Without it there is no study.**

### 5. Retention curve

For each of `{0.5, 0.35, 0.25, 0.15}`:

```bash
vitprune extract ... --out features/r025 --retention 0.25
vitprune evaluate --features features/r025 ... --out runs/r025
```

No weight is altered at any point.

### 6. Drift per retention

```bash
vitprune drift --videos data/validation ... --retention 0.25 --clips 200
```

Connects the explanation to the curve: shows how far the embedding moved, which helps
interpret the accuracy observed.

### 7. Merging against plain pruning

Compare merging with a variant that simply discards. Published methods mostly discard
without merging, so this measures directly whether preserving the pooling pays for
itself.

### 8. Visual inspection

```python
from vitprune.core import PrunedEncoder, PruningPlan

selection = PrunedEncoder(backbone, geometry, PruningPlan(retention=0.25)).select(clips)
print(selection.rows_share(clip=0, slice_index=0, rows=4))
```

Cheap: only the blocks up to the pruning layer run.

## Reporting

**Report the whole curve, not a point.**

Using the same data to choose the best retention and then to report the result at
that retention fits the evaluation set. With a small sample, that inflates the
number.

Presenting the complete curve means nothing was chosen: the trade-off is shown and
the reader picks according to their own constraint. A plot of accuracy against
latency, with five points and the baseline marked, is stronger than "3.2x for 1.4
points" and immune to that criticism.

## Measurement hygiene

**One accelerator.** Latencies from different machines are not comparable; mixing
them invalidates the curve.

**One precision throughout.** Timing in fp16 and extracting in bf16 makes the two
numbers describe different code paths.

**Record the environment.** The manifest captures versions and machine
automatically. Confirm that `evaluate` did not complain about a stride mismatch.

**Frozen models remove training-seed variance, not all variance.** GPU reductions,
ties in ranking and kernel selection can remain nondeterministic. Repeat timing
measurements and report dispersion; record the software and hardware environment.
The benchmark reports an average, not a confidence interval.

The latency sweep must start at retention `1.0`, so measured and analytic speedups
share a dense denominator. `run --compare` has no warmup and is exploratory only.
CPU execution does not apply CUDA autocast even when fp16/bf16 was requested.

## Reading realisation

Realisation is measured speedup divided by predicted.

| Value | Reading |
|---|---|
| near 100% | the operation count describes the real behaviour well |
| 50% to 80% | costs outside the count matter: sorting, merging, weight reads |
| below 50% | something dominates the time; finding out is the result |

Causes worth investigating when it is low:

- **Sorting and merging.** Real time that the multiply count ignores, and
  proportionally larger with small batches.
- **The roofline.** Reading the weights does not get cheaper with pruning.
  `vitprune cost` prints that floor and warns when it starts to dominate.
- **No fast kernel.** An operation cheap in multiplies may have no optimised
  implementation on the target hardware.

### Historical reported reference — not revalidated

The earlier documentation reports 35 clips from a 19-second video on a sixteen-core
CPU with VideoMAE Giant. Original outputs are not present in this checkout:

| | Dense | Pruned at 0.25 |
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
embeddings. Retraining it per retention ratio is cheap and removes, by construction,
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

**It multiplies the work.** One retraining per ratio, times the number of seeds.

### If you take it

Keep the no-retraining path as the comparison point. The two together answer the
question; either alone does not.
