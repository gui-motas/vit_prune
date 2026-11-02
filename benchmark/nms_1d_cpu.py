"""Pure-Python stand-in for the AMA's compiled ``nms_1d_cpu`` extension.

The compiled module in ``AMA/libs/utils`` only loads under the torch it was built
against. This is a line-by-line port of ``libs/utils/csrc/nms_cpu.cpp`` so the AMA can
run on CPU elsewhere; it is only imported when this folder comes first on the path.
"""

from __future__ import annotations

import math

import torch


def nms(segs: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if segs.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1, x2 = segs[:, 0].tolist(), segs[:, 1].tolist()
    areas = [b - a + 1e-6 for a, b in zip(x1, x2, strict=True)]
    order = scores.sort(descending=True).indices
    ranked = order.tolist()
    keep = [True] * len(ranked)
    for a, i in enumerate(ranked):
        if not keep[a]:
            continue
        for b in range(a + 1, len(ranked)):
            if not keep[b]:
                continue
            j = ranked[b]
            inter = max(0.0, min(x2[i], x2[j]) - max(x1[i], x1[j]))
            if inter / (areas[i] + areas[j] - inter) >= iou_threshold:
                keep[b] = False
    return order[torch.tensor(keep)]


def softnms(segs, scores, dets, iou_threshold, sigma, min_score, method) -> torch.Tensor:
    count = segs.shape[0]
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    x1, x2, sc = segs[:, 0].tolist(), segs[:, 1].tolist(), scores.tolist()
    areas = [b - a + 1e-6 for a, b in zip(x1, x2, strict=True)]
    inds = list(range(count))
    i = 0
    while i < count:
        best = max(range(i, count), key=lambda p: (sc[p], -p))
        dets[i, 0], dets[i, 1], dets[i, 2] = x1[best], x2[best], sc[best]
        for values in (x1, x2, sc, areas, inds):
            values[i], values[best] = values[best], values[i]
        pos = i + 1
        while pos < count:
            inter = max(0.0, min(x2[i], x2[pos]) - max(x1[i], x1[pos]))
            overlap = inter / (areas[i] + areas[pos] - inter)
            if method == 0:
                weight = 0.0 if overlap >= iou_threshold else 1.0
            elif method == 1:
                weight = 1 - overlap if overlap >= iou_threshold else 1.0
            else:
                weight = math.exp(-(overlap * overlap) / sigma)
            sc[pos] *= weight
            if sc[pos] < min_score:
                for values in (x1, x2, sc, areas, inds):
                    values[pos] = values[count - 1]
                count -= 1
                pos -= 1
            pos += 1
        i += 1
    return torch.tensor(inds[:count], dtype=torch.long)
