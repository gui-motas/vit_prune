"""Plot the FP32 and FP16 experiment: PYTHONPATH=src python benchmark/fp16_figures.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


if __name__ == "__main__":
    main()
