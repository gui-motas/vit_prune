"""Plot the FP32 and FP16 experiment: PYTHONPATH=src python benchmark/fp16_figures.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from arms_compare import false_positive_table  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

ROOT = Path("/home/gsilv257/DAR-sys/vit_prune/benchmark/results/fp16-sdpa")
OUT = Path(__file__).parent / "figuras-fp16-progressivo"
ARMS = [
    ("denso", "denso", "Denso", "Denso", "#7f7f7f"),
    ("fixo_1024_mascara", "fixo_1024_mascara", "U-1024", "U-1024 + máscara", "#ff7f0e"),
    ("prog_1536_1024_768_mascara", "prog:1536,10:1024,20:768", "Prog",
     "Progressivo 1536→1024→768 + máscara", "#2ca02c"),
]
MODES = [("a", "FP32", "latency-fp32", "o"), ("b2", "FP16+SDPA", "latency-fp16", "s")]


def action_score(arm):
    return arm["score"] if arm["recovered"] else (arm["best_any"] or {}).get("score", 0)


def plot_tokens():
    schedules = [np.full(40, 2048), np.array([2048] * 2 + [1024] * 38),
                 np.array([2048] * 2 + [1536] * 8 + [1024] * 10 + [768] * 20)]
    assert np.allclose([s.mean() for s in schedules], [2048, 1075.2, 1049.6])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    edges = np.arange(0.5, 41.5)
    for schedule, (_, _, _, label, color) in zip(schedules, ARMS, strict=True):
        mean = f"{schedule.mean():.1f}".replace(".0", "").replace(".", ",")
        ax.stairs(schedule, edges, color=color, linewidth=2, label=f"{label} (média {mean})")
        if label != "Denso":
            ax.stairs(schedule, edges, fill=True, color=color, alpha=0.15)
    for x, label in [(2.5, "corte com score + box"), (10.5, "fusão ToMe"),
                     (20.5, "fusão ToMe")]:
        ax.axvline(x, color="0.45", linestyle=":", linewidth=1)
        ax.text(x + 0.2, 200, label, rotation=90, fontsize=8, color="0.35")
    ax.set(xlim=(0.5, 40.5), ylim=(0, 2200), xlabel="Bloco", ylabel="Tokens",
           title="Tokens processados por bloco do VideoMAEv2-g (40 blocos)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", color="0.9")
    fig.tight_layout()
    fig.savefig(OUT / "tokens-por-bloco.png", dpi=150)
    plt.close(fig)


def plot_fidelity(videos):
    series = [("b2", ARMS[0], "Denso FP16+SDPA"),
              ("a", ARMS[1], "U-1024 FP32"), ("a", ARMS[2], "Progressivo FP32")]
    distances = [[] for _ in series]
    agreements = [[] for _ in series]
    for video in videos:
        data = {}
        for run, _, _, _ in MODES:
            for variant, _, _, _, _ in ARMS:
                with np.load(ROOT / run / "features" / variant / f"{video}.npz") as f:
                    data[run, variant] = (f["feats"], f["a_prob"])
        ref, ref_prob = data["a", "denso"]
        assert all(len(feats) == len(prob) == len(ref) for feats, prob in data.values())
        ref = ref.astype(np.float64)
        for i, (run, arm, _) in enumerate(series):
            feats, prob = data[run, arm[0]]
            feats = feats.astype(np.float64)
            cos = np.sum(ref * feats, axis=1) / (np.linalg.norm(ref, axis=1)
                                                  * np.linalg.norm(feats, axis=1))
            assert np.isfinite(cos).all()
            distances[i].extend(1 - cos)
            agreements[i].extend(ref_prob.argmax(axis=1) == prob.argmax(axis=1))
    assert all(len(d) == 2763 for d in distances)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    summaries = []
    clipped = 0
    for (_, arm, label), values, matches in zip(series, distances, agreements, strict=True):
        values = np.asarray(values)
        cos = 1 - values
        median = np.median(cos)
        p10 = np.percentile(cos, 10)
        agreement = np.mean(matches) * 100
        clipped += np.count_nonzero(values < 1e-8)
        assert np.all(values <= 1)
        ax.hist(np.maximum(values, 1e-8), bins=np.logspace(-8, 0, 81), histtype="step",
                color=arm[4], linewidth=1.8, label=label)
        ax.axvline(max(1 - median, 1e-8), color=arm[4], linestyle="--", linewidth=1)
        summaries.append(f"{label}: top-1 {agreement:.2f}% · {len(values)} clipes\n"
                         f"mediana cos = {median:.6f}")
        print(f"Fidelidade {label}: mediana cos={median:.9f}; P10 cos={p10:.9f}; "
              f"top-1={agreement:.2f}%; clipes={len(values)}")
    ax.set_xscale("log")
    ax.set(xlim=(1e-8, 1), xlabel="1 − cosseno com o denso FP32 (escala log)",
           ylabel="Número de clipes por intervalo",
           title="Distância ao embedding denso FP32 por clipe (2.763 clipes): precisão × poda")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.52, 0.97, "\n".join(summaries), transform=ax.transAxes, ha="center", va="top",
            fontsize=8, bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
    if clipped:
        fig.text(0.5, 0.01, f"{clipped} distâncias abaixo de 10⁻⁸ truncadas em 10⁻⁸.",
                 ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04 if clipped else 0, 1, 1))
    fig.savefig(OUT / "fidelidade-fp16.png", dpi=150)
    plt.close(fig)


def plot_scores(actions):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (run, title) in zip(axes, [("a", "FP32 + atenção original"),
                                       ("b2", "FP16 + SDPA")], strict=True):
        fixed = [a["arms"][f"{run}/{ARMS[1][0]}"] for a in actions]
        prog = [a["arms"][f"{run}/{ARMS[2][0]}"] for a in actions]
        x = np.array([action_score(a) for a in fixed])
        y = np.array([action_score(a) for a in prog])
        dense = np.array([action_score(a["arms"][f"{run}/denso"]) for a in actions])
        recovered = np.array([u["recovered"] and p["recovered"]
                              for u, p in zip(fixed, prog, strict=True)])
        diff = y[recovered] - x[recovered]
        median, greater = np.median(diff), np.count_nonzero(diff > 0)
        print(f"Scores {title}: mediana Prog−U={median:.6f}; "
              f"Prog>U={greater}/{len(diff)}; perdidas={len(actions) - len(diff)}")
        ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.8)
        ax.axvline(0.30, color="0.5", linestyle="--", linewidth=0.8)
        ax.axhline(0.30, color="0.5", linestyle="--", linewidth=0.8)
        points = ax.scatter(x[recovered], y[recovered], c=dense[recovered], cmap="viridis",
                            vmin=0, vmax=1, s=35, zorder=3)
        ax.scatter(x[~recovered], y[~recovered], facecolors="none", edgecolors="red",
                   s=55, linewidths=1.4, zorder=4, label="ação perdida")
        for video, label, offset in [("13522_NoAudio_7", 6, (8, 30)),
                                     ("33682_NoAudio_5", 8, (-90, 15))]:
            indices = [i for i, action in enumerate(actions)
                       if video in action["video-id"] and action["label"] == label]
            assert len(indices) == 1
            i = indices[0]
            short_video = video.replace("_NoAudio", "")
            ax.annotate(f"{short_video} classe {label}", (x[i], y[i]), xytext=offset,
                        textcoords="offset points", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="0.3"))
        ax.text(0.03, 0.97, f"Mediana Prog − U = {median:.3f}\nProg > U em {greater}/{len(diff)}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
        ax.text(0.31, 0.02, "corte 0,30", fontsize=8, color="0.4")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel="Score U-1024")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Score Progressivo")
    fig.suptitle("Score do AMA por ação anotada: U-1024 × Progressivo (48 ações)")
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.12, top=0.85, wspace=0.12)
    fig.colorbar(points, cax=fig.add_axes([0.89, 0.17, 0.018, 0.63]), label="score do denso")
    fig.savefig(OUT / "scores-pareados.png", dpi=150)
    plt.close(fig)


def main():
    comparison = json.loads((ROOT / "comparacao.json").read_text())
    actions = comparison["actions"]
    gt = pd.DataFrame(actions)
    latencies = {run: json.loads((ROOT / folder / "latency.json").read_text())["arms"]
                 for run, _, folder, _ in MODES}
    detections = {f"{run}/{variant}": pd.read_csv(ROOT / run / "ama" / variant / "predictions.csv")
                  for run, _, _, _ in MODES for variant, _, _, _, _ in ARMS}
    indexed = {key: frame.assign(detection_index=frame.index) for key, frame in detections.items()}
    fps = false_positive_table(gt, indexed)
    fp_indices = {key: set(fps.loc[fps["arm"] == key, "detection_index"])
                  for key in detections}

    print("Arm | Mode | p50 ms | Recovered | FP | mAP")
    for run, mode, _, _ in MODES:
        for variant, latency_key, name, _, _ in ARMS:
            key = f"{run}/{variant}"
            arm = comparison["arms"][key]
            ms = latencies[run][latency_key]["p50_ms"]
            print(f"{name} | {mode} | {ms:.0f} | {arm['recovered']} | "
                  f"{arm['false_positives']} | {arm['map']:.3f}")
            assert len(fp_indices[key]) == arm["false_positives"]
            assert sum(a["arms"][key]["recovered"] for a in actions) == arm["recovered"]
    assert len(actions) == 48
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, metric, ylabel, ref in zip(
        axes, ("recovered", "map"),
        ("Ações anotadas reencontradas (de 48)", "mAP médio (tIoU 0,1–0,5)"),
        (48, 1.0), strict=True,
    ):
        ax.axhline(ref, color="gray", linestyle="--", linewidth=0.8, zorder=0)
        for index, (variant, latency_key, name, _, color) in enumerate(ARMS):
            xs = [latencies[run][latency_key]["p50_ms"] for run, _, _, _ in MODES]
            ys = [comparison["arms"][f"{run}/{variant}"][metric] for run, _, _, _ in MODES]
            ax.plot(xs, ys, color=color, linestyle=":", linewidth=1)
            for mode_index, (run, mode, _, marker) in enumerate(MODES):
                arm = comparison["arms"][f"{run}/{variant}"]
                x, y = xs[mode_index], ys[mode_index]
                ax.scatter(x, y, marker=marker, s=60, edgecolors=color,
                           facecolors="none" if mode_index == 0 else color, zorder=3)
                offset = [(8, 8), (8, -29), (-8, 9)][index]
                if index == 0 and mode_index == 0:
                    offset = (-8, 8)
                if index == 2 and mode_index == 1:
                    offset = (8, 9)
                ax.annotate(f"{name} {mode}\n{x:.0f} ms · {arm['false_positives']} FP",
                            (x, y), xytext=offset, textcoords="offset points", fontsize=7,
                            ha="right" if offset[0] < 0 else "left", color=color)
        ax.set_xscale("log")
        ax.set_xlim(130, 2500)
        ax.xaxis.set_major_locator(FixedLocator([150, 250, 500, 1000, 2000]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Latência p50 por clipe (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", color="0.9", linewidth=0.7)
    axes[0].set_ylim(44.6, 49.2)
    axes[1].set_ylim(0.963, 1.01)
    handles = [Line2D([], [], color=color, lw=2, label=label) for _, _, _, label, color in ARMS]
    handles += [Line2D([], [], color="black", marker=marker, linestyle="none",
                       markerfacecolor="none" if run == "a" else "black", label=label)
                for run, label, marker in (("a", "FP32 + atenção original", "o"),
                                           ("b2", "FP16 + SDPA", "s"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Fronteira latência × qualidade — 3 vídeos de dev, 48 ações "
                 "(Jetson AGX Orin, lote 8)")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    fig.savefig(OUT / "fronteira.png", dpi=150)
    plt.close(fig)

    videos = sorted(gt["video-id"].unique())
    assert len(videos) == 3
    fig, axes = plt.subplots(3, 1, figsize=(18, 11))
    colors = plt.get_cmap("tab20").colors
    counts = {key: [0, 0] for key in detections}
    for ax, video in zip(axes, videos, strict=True):
        video_actions = [a for a in actions if a["video-id"] == video]
        for action in video_actions:
            ax.barh(0, action["t-end"] - action["t-start"], left=action["t-start"],
                    height=0.7, color=colors[int(action["label"])])
        last_end = max(a["t-end"] for a in video_actions)
        for row, (run, _, _, _) in enumerate(MODES):
            for index, (variant, _, _, _, _) in enumerate(ARMS):
                key = f"{run}/{variant}"
                y = 1 + row * 3 + index
                frame = detections[key]
                frame = frame[frame["video-id"] == video]
                if not frame.empty:
                    last_end = max(last_end, frame["t-end"].max())
                for detection_index, detection in frame[frame["score"] >= 0.30].sort_values(
                    "score"
                ).iterrows():
                    is_fp = detection_index in fp_indices[key]
                    counts[key][1] += is_fp
                    ax.barh(y, detection["t-end"] - detection["t-start"],
                            left=detection["t-start"], height=0.7,
                            color=(*colors[int(detection["label"])][:3],
                                   0.35 + 0.65 * min(detection["score"], 1)),
                            edgecolor="black" if is_fp else "none", linewidth=2 if is_fp else 0)
                for action in video_actions:
                    if not action["arms"][key]["recovered"]:
                        counts[key][0] += 1
                        ax.plot((action["t-start"] + action["t-end"]) / 2, y,
                                marker="x", color="red", markersize=10, markeredgewidth=2,
                                zorder=5)
        ax.axhline(3.5, color="gray", linewidth=0.6)
        ax.set_yticks(range(7), ["Anotações"] + [f"{name} {mode}"
                      for _, mode, _, _ in MODES for _, _, name, _, _ in ARMS])
        ax.set_ylim(6.6, -0.6)
        ax.set_xlim(0, last_end + 10)
        ax.set_xlabel("segundos no vídeo")
        ax.set_title(video.removeprefix("Dashboard_user_id_"))
        ax.grid(axis="x", color="0.9", linewidth=0.6)
    legend = [Patch(color=colors[i], label=f"classe {i}") for i in range(16)]
    legend += [Patch(facecolor="white", edgecolor="black", linewidth=2,
                     label="falso positivo (contorno preto)"),
               Line2D([], [], color="red", marker="x", linestyle="none", markersize=8,
                      label="ação perdida (✗)")]
    fig.legend(handles=legend, loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle("Detecções do AMA (confiança ≥ 0,30) por braço — anotações, FP32 e FP16+SDPA")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(OUT / "linha-do-tempo.png", dpi=130)
    plt.close(fig)

    print("Arm | Mode | missed x | FP outlines")
    for run, mode, _, _ in MODES:
        for variant, _, name, _, _ in ARMS:
            key = f"{run}/{variant}"
            missed, outlined = counts[key]
            assert missed == len(actions) - comparison["arms"][key]["recovered"]
            assert outlined == comparison["arms"][key]["false_positives"]
            print(f"{name} | {mode} | {missed} | {outlined}")

    plot_tokens()
    plot_fidelity(videos)
    plot_scores(actions)


if __name__ == "__main__":
    main()
