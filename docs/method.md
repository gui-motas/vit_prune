# How the method works

Terms are defined where they first appear. The short version is in the README; this
is the reasoning behind each decision.

## 1. A clip becomes visual tokens

The backbone does not process a whole video. It processes a **clip**: a fixed number
of frames, typically sixteen, covering about half a second.

Each frame is resized and cut into square **patches**. With a 224-pixel frame and a
patch size of 14, that is 16 by 16 = 256 patches per frame.

The patch embedding is three-dimensional: it also groups neighbouring frames into
**tubelets**. With a tubelet size of 2, sixteen frames collapse into eight
**temporal slices**.

The result is `8 slices x 256 patches = 2048 visual tokens` per clip. Each token is a
vector describing one region of the image at one moment.

## 2. Why the cost is what it is

A large video transformer has on the order of forty blocks. In every block,
self-attention compares each token against all the others: with 2048 tokens that is
roughly 4.2 million pairs, repeated forty times.

The cost of attention grows with the square of the token count. Doubling the tokens
quadruples it. The reverse also holds, and that is the opening.

## 3. The redundancy being exploited

With a fixed camera, most of those tokens carry nothing. The ceiling is always the
same ceiling; the seat back does not move; the dashboard is static. Hundreds of
near-identical tokens take part in all 4.2 million pairs, in all forty blocks.

The evidence of an action sits in a handful of tokens: a hand, a face, an object.

So: keep those and drop the rest. That is the whole idea.

## 4. Deciding which tokens to keep

Two criteria, combined.

### Uniqueness

How far a token sits from the rest of the scene, measured by cosine similarity
against a mean. A ceiling token resembles hundreds of others and scores low. A hand
holding an object resembles nothing nearby and scores high.

The comparison runs at two levels: against the mean of the **whole clip**, which
catches what is rare overall, and against the mean of the token's **own temporal
slice**, which catches what is rare at that instant. The two contributions are added;
this favours deviation at both levels without requiring both to be large.

### Motion

The L2 distance between the embedding at one position and the embedding at the same
position in the previous slice. If the content there changed, something moved. Motion
is what carries evidence of an action rather than of a scene.

### Combining them

The two are combined through a **weighted geometric mean**, not a sum. With a sum, a
token scoring high on one and zero on the other would still pass. With a product it
has to do reasonably well on both, which rules out what is merely unusual but static,
and what merely flickers without being relevant.

The weight is written as η in the literature and `motion_weight` here; 0.5 balances
the two. It stays below 1: the first slice has no motion, so under motion alone its tokens
would tie and be kept in grid order, whatever the clip shows.

Both scores are normalised to a common range before combining, because uniqueness is
a cosine between -1 and 1 while motion is an unbounded distance. Normalisation is per
slice, so that one busy slice cannot flatten the scores everywhere else.

### How many tokens each clip keeps

The score decides *which* tokens survive. *How many* is decided per clip, because
clips differ: a still, repetitive scene can lose most of its tokens, while one with
motion and detail needs more of them.

Two clip-level statistics are measured on the same grid, before any merge:
**redundancy**, the mean cosine similarity of the tokens to the clip mean and to their
slice mean, and **motion**, the mean distance between the same position in
consecutive slices. Neither is normalised per slice: that would erase exactly the
differences between clips the budget has to see.

Both have no natural scale, so each is turned into its rank among reference values
measured on **calibration clips**, which must not be part of the evaluation. The ranks
combine into `g = β·motion + (1 − β)·(1 − redundancy)`: more motion and less
redundancy ask for more tokens. Thresholds on `g` sort clips into a few **regimes**,
each with a fixed token count, by default 256, 512 and 768. Calibration places the
thresholds so the regimes are equally frequent on the calibration clips, which puts
the mean at 512 tokens.

Each slice of a clip keeping 512 tokens keeps its 64 highest-scoring tokens, out of
256; the pairs per block fall from 4.2 million to 262 thousand, sixteen times fewer.
A few regimes rather than a continuous count let a batch run as at most one
sub-batch per regime, without padding.

## 5. Why dropping tokens is not enough

This is the part that decides whether the method is usable at all.

The backbone produces the **clip embedding** by mean pooling over its tokens. That
embedding is what a downstream model consumes.

If pruning simply discarded the other tokens, the mean would change meaning. Only the
salient regions would remain, so the embedding would describe a crop rather than a
scene. And a downstream model trained on dense embeddings expects the old meaning.

### Token merging

Instead of discarding, each pruned token is assigned to the kept token with the
highest cosine similarity. That kept token then carries a **size**: it stands for
twelve tokens rather than one.

At the final pooling, each token counts according to its size. An exact identity
follows: **size-weighted pooling over the survivors equals mean pooling over all the
original tokens.** There are fewer tokens and the pooled mean is unchanged.

Similarity ignores magnitude, which is deliberate: two ceiling tokens point the same
way even if one is better lit, and that is the equivalence the merge exploits.

### The honest caveat

The identity holds **at the moment of the merge**. The remaining blocks keep
transforming the surviving tokens, which are not the tokens that would have existed
without merging, so the two paths separate. That separation is **drift**, and
`vitprune drift` measures it rather than assuming it is small.

## 6. Which layer to prune at

Pruning happens after the second of forty blocks. Two opposing pressures set that.

**Earlier tokens have less context.** This may weaken selection quality; the size
of that effect is an empirical question, not a proven impossibility of earlier cuts.

**Pruning later wastes.** Cutting at block thirty means having paid full cost for
thirty blocks in order to save on ten.

Two blocks are the default hypothesis, leaving thirty-eight blocks to save on.
Compare cuts empirically rather than treating this as a universal optimum.

An optional progressive schedule makes later ToMe-style merges after the initial cut.
Its quality and latency must be measured against the single-cut baseline.

## 7. Spatial priors, and when not to use them

Both criteria are blind to meaning. In a vehicle-mounted camera that has a concrete
consequence: the windscreen scores very high, being at once the region of greatest
motion and greatest uniqueness. Budget goes to the view out of the window, which is
where the information sought is not.

A per-position multiplier lowers the score of known bands of the frame.

**But check before using it.** In one inspection of driver-facing footage the method
spent only 2% of its budget on the top four rows, which occupy 25% of the frame. In
that material the top was the roof lining -- uniform and static -- and the window sat
on the left edge. Enabling a top-rows prior would have penalised a region already
being correctly ignored.

A prior encodes where the camera was installed, not a property of the method. Use
`TokenSelection` to look at a real frame first.

**The driver region takes the band from data instead.** `vitprune region` follows the
AI City preprocessing, which crops each video to the largest confident person box in it
(COCO person above 0.5). It runs a COCO detector (YOLO26s) on the middle frame of every
calibration clip, keeps each video's largest box as that video's crop, and takes the
median of the crops, coordinate by coordinate, as the region: one rectangle, like the
crops the downstream model was trained on.
Joining every box instead fails with a dashboard camera: the driver fills about 40% of the
frame and moves, and the union over 16 calibration videos covered 91% of the frame.
Positions inside get weight 1, the rest 0: the slots of each slice go first to the
region, by score, and outside tokens only take what is left, still merging into kept
ones. The budget statistics stay on the whole grid, so the region changes which tokens
a clip keeps, never how many. Whether that helps is the experiment in
[experiments](experiments.md#validating-the-driver-region): every outside token merges
into a token of the region, even when nothing there resembles it.

## 8. Proportional attention

A kept token standing for twelve should arguably be attended to twelve times as
strongly. Adding `log(size)` to the attention logits would achieve that, but forces
the logit matrix into memory -- exactly what efficient attention kernels avoid, which
would be an odd trade in a method about speed.

The alternative appends one dimension: a constant to the query and `log(size)` to the
key. Their inner product contributes the same term with nothing materialised.

This is off by default. The pooling identity does not depend on it. Enabling it is a
hypothesis to test.

## 9. What the method does not do

- **It does not train.** No parameter is updated anywhere.
- **It is not query-aware.** Pruning methods for video language models use text
  attention to select tokens relevant to a question. There is no question here.
- **It does not segment scenes.** Methods that do operate on long, multi-shot video.
  A clip of eight temporal slices from a single fixed shot has nothing to segment.

## 10. Summary

1. A clip becomes 2048 visual tokens
2. Self-attention is quadratic, and runs in every block
3. After two blocks, the clip's redundancy and motion decide how many tokens it keeps
4. Each token is scored by uniqueness and by motion; the highest scoring survive and
   the rest are merged into their nearest match, carrying their size forward
5. The remaining blocks run on the reduced tokens, one sub-batch per regime
6. Size-weighted pooling honours merged token masses; final embeddings can drift

With the default levels the clips average 512 tokens, but the expected cost is higher
than that of 512 tokens in every clip: attention is quadratic, so the cost has to come
from the counts the clips actually kept. Analytic predictions and clocks are
different things, and the distance between them is what
[`experiments.md`](experiments.md) is about.
