"""Visualize real VideoMAE pruning and merging on two annotated dev clips.

Run with PYTHONPATH=src python benchmark/pruning_visual.py --videos DIR --out DIR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from c2_check import setup
from PIL import Image

from vitprune.bench.latency import autocast
from vitprune.core import FixedBudget, PrunedEncoder, PruningPlan
from vitprune.core.merging import (
    _bipartite_matches,
    _nearest_kept_token,
    _take,
    bipartite_merge,
    initial_sizes,
    merge_tokens,
)
from vitprune.core.scoring import score_tokens
from vitprune.pipeline.benchmark.inference import video_info
from vitprune.pipeline.benchmark.settings import GEOMETRY
from vitprune.pipeline.video import find_videos, iter_clips, read_frames

ROOT = Path("/home/gsilv257/DAR-sys")
DEFAULT_VIDEOS = ROOT / "vit_prune/benchmark/results/fp16-sdpa/sep/videos"
DEFAULT_REGION = ROOT / "vit_prune/benchmark/results/qabp/regiao/regiao-ajuste.json"
DEFAULT_WEIGHTS = ROOT / "models-weights/videomaev2/videomae-v2_inference.pth"
DEFAULT_SOURCE = ROOT / "models-weights/AIcity2024-track3/feature_extraction"
CLIPS = (
    ("Dashboard_user_id_31903_NoAudio_7", "31903-7", 479.5, 6, "469–490"),
    ("Dashboard_user_id_13522_NoAudio_7", "13522-7", 132.0, 5, "122–142"),
)
STAGES = ((10, 1024), (20, 768))


def stage_map(provenance, unmerged, source, destination, count):
    """Map original positions from old token indices to bipartite output indices."""
    lookup = torch.empty(count, dtype=torch.long, device=provenance.device)
    n = unmerged.numel()
    lookup[2 * unmerged] = torch.arange(n, device=lookup.device)
    lookup[2 * source] = n + destination
    lookup[1::2] = n + torch.arange(count // 2, device=lookup.device)
    return torch.where(provenance >= 0, lookup[provenance.clamp(min=0)], -1)


def snapshot(provenance, identities, sizes):
    valid = provenance >= 0
    counts = torch.bincount(provenance[valid], minlength=identities.numel())
    assert torch.equal(counts, sizes.reshape(-1).long()), "provenance differs from token sizes"
    ids = torch.where(valid, identities[provenance.clamp(min=0)], -1)
    return ids.reshape(8, 256).cpu().numpy()


@torch.inference_mode()
def trace(encoder, clip):
    """Reproduce _encode for a single clip; save an original-to-representative map."""
    plan = encoder.plan
    blocks = list(encoder.backbone.blocks)
    tokens = encoder._tokenise(clip)
    encoder._check_grid(tokens.shape[1])
    for block in blocks[:plan.prune_layer]:
        tokens = encoder._run(block, tokens, None)
    grid = encoder._as_grid(tokens, GEOMETRY.tokens_per_slice)
    base = score_tokens(grid, plan.motion_weight)
    keeps, _ = encoder._keeps(grid, base)
    keep = int(keeps[0])
    scores = encoder._with_prior(base)
    positions = torch.arange(256, device=grid.device)
    if plan.discard_outside:
        inside = plan.spatial_prior.to(grid.device) > 0
        grid, scores, positions = grid[:, :, inside], scores[:, :, inside], positions[inside]
    _, slices, total, dim = grid.shape
    ranking = scores.argsort(dim=-1, descending=True)
    kept_at = ranking[..., :keep].sort(dim=-1).values
    kept_positions = positions[kept_at[0]]
    kept_mask = torch.zeros((slices, 256), dtype=torch.bool, device=grid.device)
    kept_mask.scatter_(1, kept_positions, True)
    provenance = torch.full((slices, 256), -1, dtype=torch.long, device=grid.device)
    offsets = torch.arange(slices, device=grid.device)[:, None] * keep
    identities = (kept_positions + torch.arange(slices, device=grid.device)[:, None] * 256)
    identities = identities.flatten()
    if keep == total:
        group = grid.reshape(1, slices * keep, dim)
        sizes = initial_sizes(grid).float() if plan.discard_outside else None
        provenance[:, positions] = offsets + torch.arange(keep, device=grid.device)
    else:
        sizes = initial_sizes(grid)
        if plan.stages or plan.discard_outside:
            sizes = sizes.float()
        pruned_at = ranking[..., keep:]
        destinations = _nearest_kept_token(_take(grid, pruned_at), _take(grid, kept_at))[0]
        assignment = torch.empty((slices, total), dtype=torch.long, device=grid.device)
        own = torch.arange(keep, device=grid.device).expand(slices, -1)
        assignment.scatter_(1, kept_at[0], own)
        assignment.scatter_(1, pruned_at[0], destinations)
        provenance[:, positions] = offsets + assignment
        merged = merge_tokens(grid, sizes, scores, keep)
        group, sizes = merged.tokens.to(grid.dtype).reshape(1, slices * keep, dim), merged.sizes

    maps = {2: snapshot(provenance, identities, sizes)}
    for index, block in enumerate(blocks[plan.prune_layer:], start=plan.prune_layer + 1):
        group = encoder._run(block, group, sizes)
        if index in dict(plan.stages):
            sizes = sizes.reshape(1, -1, 1)
            metric = encoder._stage_keys(blocks[index], group)
            target = dict(plan.stages)[index]
            unmerged, source, destination = _bipartite_matches(metric, target)
            unmerged, source, destination = (v[0, :, 0] for v in (unmerged, source, destination))
            provenance = stage_map(provenance, unmerged, source, destination, group.shape[1])
            identities = torch.cat((identities[::2][unmerged], identities[1::2]))
            group, sizes, _ = bipartite_merge(group, sizes, metric, target)
        if index in (10, 20):
            maps[index] = snapshot(provenance, identities, sizes)
    return encoder._pool(group, sizes), kept_mask.cpu().numpy(), maps


def group_stats(ids):
    live = ids[ids >= 0]
    names, counts = np.unique(live, return_counts=True)
    spanning = sum(np.unique(np.where(ids == name)[0]).size > 1 for name in names)
    return (spanning, float(counts.mean()), int(counts.max()),
            float(counts[counts > 1].sum() / live.size))


def get_clip(bench, stem, middle):
    videos = [p for p in find_videos(bench.videos) if p.stem == stem]
    if len(videos) != 1:
        raise ValueError(f"expected exactly one dev video named {stem}, found {len(videos)}")
    video = videos[0]
    info = video_info(video)
    if info["fps"] <= 0:
        raise ValueError(f"invalid frame rate: {video}")
    start = min(bench.clip_starts(info["frames"]),
                key=lambda s: abs((s + bench.span / 2) / info["fps"] - middle))
    decoded = dict(iter_clips(video, GEOMETRY, 16, 2, [start]))
    if start not in decoded:
        raise ValueError(f"scheduled clip {start} could not be decoded from {video}")
    frames = []
    for offset, frame in enumerate(read_frames(video, start)):
        if offset % 4 == 0:
            frames.append(frame)
        if offset == 28:
            break
    if len(frames) != 8:
        raise ValueError(f"not enough display frames in {video}")
    return start, info["fps"], decoded[start].unsqueeze(0), frames


def draw_frame(ax, frame, rgba=None, kept=None):
    height, width = frame.shape[:2]
    ax.imshow(frame, extent=(0, 16, 16, 0), aspect="auto")
    if rgba is not None:
        ax.imshow(rgba.reshape(16, 16, 4), extent=(0, 16, 16, 0),
                  interpolation="nearest", aspect="auto")
    for p in range(17):
        ax.axhline(p, color="white", alpha=.28, linewidth=.35)
        ax.axvline(p, color="white", alpha=.28, linewidth=.35)
    if kept is not None:
        from matplotlib.collections import LineCollection
        lines = []
        for r, c in np.argwhere(kept.reshape(16, 16)):
            lines.append([(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1), (c, r)])
        ax.add_collection(LineCollection(lines, colors="#5af47b", linewidths=.42, alpha=.9))
    ax.set_xlim(0, 16)
    ax.set_ylim(16, 0)
    ax.set_box_aspect(height / width)
    ax.axis("off")


def selection_overlay(kept, discard=False):
    rgba = np.zeros((256, 4))
    rgba[~kept] = (0.32, 0.02, 0.02, .88) if discard else (0, 0, 0, .66)
    return rgba


def merge_overlay(ids, temporal, colors):
    counts = np.bincount(ids[ids >= 0], minlength=2048)
    rgba = np.zeros((256, 4))
    for i, identity in enumerate(ids[temporal]):
        if identity < 0:
            rgba[i] = (0, 0, 0, .96)
        else:
            rgba[i, :3] = colors[identity]
            rgba[i, 3] = 1 if counts[identity] > 1 else .18
    return rgba


def save(fig, path):
    fig.savefig(path, dpi=110, facecolor="white", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    # Palette PNGs keep the thesis figures small without storing temporary copies.
    with Image.open(path) as image:
        image.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
            path, optimize=True)
    print(f"{path}: {path.stat().st_size / 1024:.0f} KiB", flush=True)


def make_box(out, frame, prior):
    fig, ax = plt.subplots(figsize=(8, 4.7))
    overlay = np.zeros((256, 4))
    overlay[prior] = (1, 0, .7, .23)
    draw_frame(ax, frame, overlay)
    ax.set_title("Região do motorista (box): 169 de 256 posições por fatia", fontsize=12)
    save(fig, out / "poda-box.png")


def make_cut(out, short, frames, results, prior, title):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    draw_frame(axes[0], frames[4])
    axes[0].set_title("Quadro original\n(fatia t=4)")
    for ax, name in zip(axes[1:], ("U-1024", "P", "P-drop"), strict=True):
        kept = results[name][0][4]
        inside = 100 * (kept & prior).sum() / kept.sum()
        drop = name == "P-drop"
        draw_frame(ax, frames[4], selection_overlay(kept, drop), kept)
        detail = "87 descartados" if drop else f"{256 - kept.sum()} fundidos"
        ax.set_title(f"{name} — corte: {kept.sum()} mantidos\n"
                     f"{detail}; {inside:.1f}% dentro da box")
    fig.suptitle(title, fontsize=11)
    fig.text(.5, .02, "Verde: mantido  •  escuro: fundido  •  vermelho-escuro: descartado",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .9))
    save(fig, out / f"poda-corte-{short}.png")


def make_slices(out, short, frames, kept, name, discard):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for t, ax in enumerate(axes.flat):
        draw_frame(ax, frames[t], selection_overlay(kept[t], discard),
                   None if discard else kept[t])
        ax.set_title(f"Fatia t={t}", fontsize=10)
    text = "vermelho-escuro: descartado" if discard else "verde: mantido; escuro: fundido"
    fig.suptitle(f"{name}: seleção nas oito fatias temporais — {text}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.2)
    suffix = "" if discard else "-u1024"
    save(fig, out / f"poda-fatias{suffix}-{short}.png")


def make_tome(out, short, frames, maps):
    colors = np.random.default_rng(42).uniform(.15, 1, (2048, 3))
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 6)
    positions = [(0, 0, 2, 4), (0, 2, 10, 4), (0, 4, 20, 4),
                 (1, 1, 20, 2), (1, 3, 20, 6)]
    for row, col, stage, temporal in positions:
        ax = fig.add_subplot(grid[row, col:col + 2])
        draw_frame(ax, frames[temporal], merge_overlay(maps[stage], temporal, colors))
        label = {2: "após o corte (bloco 2): 1352 tokens",
                 10: "após o bloco 10: 1024 tokens",
                 20: "após o bloco 20: 768 tokens"}[stage]
        ax.set_title(f"{label}\nfatia t={temporal}", fontsize=10)
    fig.suptitle("P-drop: cada cor identifica o token representante em todas as fatias",
                 fontsize=12)
    fig.text(.5, .02, "Cor forte: grupo com 2 ou mais posições; "
             "cor leve: isolado; preto: descartado",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, out / f"poda-tome-{short}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="pasta de vídeos dev")
    parser.add_argument("--out", type=Path, default=Path("benchmark/figuras-fp16-progressivo"))
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    # CUDA's FP16 scatter-add is otherwise nondeterministic between the two forwards.
    torch.use_deterministic_algorithms(True)
    print("FP16/SDPA; acumulação CUDA determinística para a autoverificação", flush=True)
    bench, model, region = setup(args)
    prior = (region > 0).numpy()
    if prior.sum() != 169 or GEOMETRY.temporal_slices != 8:
        raise ValueError("expected a box of 169 positions and eight temporal slices")
    plans = {
        "U-1024": PruningPlan(prune_layer=2, budget=FixedBudget(1024), spatial_prior=region,
                              attention="sdpa"),
        "P": PruningPlan(prune_layer=2, budget=FixedBudget(1536), spatial_prior=region,
                         stages=STAGES, attention="sdpa"),
        "P-drop": PruningPlan(prune_layer=2, budget=FixedBudget(1352), spatial_prior=region,
                              stages=STAGES, attention="sdpa", discard_outside=True),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for stem, short, middle, action, interval in CLIPS:
        start, fps, clip, frames = get_clip(bench, stem, middle)
        print(f"{stem}: início={start}, fps={fps:g}, centro={(start + 16) / fps:.3f}s", flush=True)
        clip = clip.to(bench.device)
        results = {}
        for name, plan in plans.items():
            encoder = PrunedEncoder(model, GEOMETRY, plan)
            with torch.inference_mode(), autocast(bench.device, "fp16"):
                embedding, kept, maps = trace(encoder, clip)
                reference = encoder.encode(clip)
            delta = float((embedding.float() - reference.float()).abs().max())
            print(f"{short} {name}: max |Δ| = {delta:.9g}", flush=True)
            if not np.isfinite(delta) or delta > 1e-3:
                raise AssertionError(f"embedding self-check failed for {short} {name}: {delta}")
            results[name] = (kept, maps)
        for stage in (10, 20):
            ids = results["P-drop"][1][stage]
            spanning, mean, maximum, fraction = group_stats(ids)
            original_fraction = fraction * np.count_nonzero(ids >= 0) / ids.size
            print(f"{short} P-drop bloco {stage}: grupos entre fatias={spanning}; "
                  f"tamanho médio={mean:.3f}, máximo={maximum}; "
                  f"posições em grupos múltiplos={fraction:.2%} das mantidas "
                  f"({original_fraction:.2%} das originais)", flush=True)
        title = (f"{stem} — centro {(start + 16) / fps:.2f} s — "
                 f"ação anotada classe {action} ({interval} s)")
        make_cut(args.out, short, frames, results, prior, title)
        if short == CLIPS[0][1]:
            make_box(args.out, frames[4], prior)
            make_slices(args.out, short, frames, results["P-drop"][0], "P-drop", True)
            make_slices(args.out, short, frames, results["U-1024"][0], "U-1024", False)
            make_tome(args.out, short, frames, results["P-drop"][1])


if __name__ == "__main__":
    main()
