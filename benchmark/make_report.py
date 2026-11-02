"""Turn the benchmark records into a spreadsheet and figures.

Reads ``results/clips.jsonl`` and the files next to it, and writes:

* ``results/benchmark-vitprune.xlsx`` -- summary, per class, per video, per clip and the
  configuration the numbers were produced under;
* ``results/figuras/`` -- the masks over a frame of each video, one panel per action with
  the token selection of every variant, and a chart of the summary.

Can run while ``run_inference.py`` is still going; it reports what is there.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import CACHE, GEOMETRY, RESULTS, segments, videos  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from run_inference import OLD_MASK, PRUNE_LAYER  # noqa: E402

from vitprune.bench.cost import budget_cost  # noqa: E402

NAMES = {
    "denso": "Sem ViT-Prune (denso)",
    "vitprune": "ViT-Prune sem máscara",
    "vitprune_mascara_antiga": "ViT-Prune + máscara antiga",
    "vitprune_regiao": "ViT-Prune + região do motorista",
}
SHORT = {
    "denso": "Denso",
    "vitprune": "Sem máscara",
    "vitprune_mascara_antiga": "Máscara antiga",
    "vitprune_regiao": "Região",
}
FIGURES = RESULTS / "figuras"
GRID = GEOMETRY.grid_size
MIDDLE_SLICE = GEOMETRY.temporal_slices // 2


def load() -> pd.DataFrame:
    rows = [json.loads(line) for line in (RESULTS / "clips.jsonl").read_text("utf-8").splitlines()]
    frame = pd.DataFrame(rows)
    complete = frame.groupby(["video", "start_frame"])["variant"].transform("count") == len(NAMES)
    frame = frame[complete].copy()
    # Clips outside every action carry no label: NaN keeps them out of the accuracies.
    for column in ("correct", "top3"):
        frame[column] = frame[column].astype(float)
    names = {s["label_id"]: s["label"] for v in videos() for s in segments(v.stem)}
    frame["predicted"] = frame["predicted_id"].map(names)
    dense = frame[frame["variant"] == "denso"].set_index(["video", "start_frame"])["predicted_id"]
    frame["agrees_with_dense"] = [
        dense[(v, s)] == p for v, s, p in zip(frame["video"], frame["start_frame"],
                                              frame["predicted_id"], strict=True)
    ]
    return frame


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    dense_seconds = frame[frame["variant"] == "denso"]["seconds"].mean()
    table = []
    for variant, rows in frame.groupby("variant", sort=False):
        per_class = rows.groupby("label_id")["correct"].mean()
        tokens = rows["tokens"].astype(int).tolist()
        layer = 0 if variant == "denso" else PRUNE_LAYER
        table.append({
            "Variante": NAMES[variant],
            "Clipes": len(rows),
            "Clipes com rótulo": int(rows["correct"].notna().sum()),
            "Acurácia top-1": rows["correct"].mean(),
            "Acurácia top-3": rows["top3"].mean(),
            "Acurácia balanceada": per_class.mean(),
            "Concordância com o denso": rows["agrees_with_dense"].mean(),
            "Confiança média": rows["confidence"].mean(),
            "Cosseno médio com o denso": rows["cosine_to_dense"].mean(),
            "Cosseno p5 com o denso": rows["cosine_to_dense"].quantile(0.05),
            "Tokens médios por clipe": np.mean(tokens),
            "Speedup previsto": budget_cost(GEOMETRY, tokens, layer).predicted_speedup,
            "Tempo médio por clipe (s)": rows["seconds"].mean(),
            "Speedup medido": dense_seconds / rows["seconds"].mean(),
        })
    return pd.DataFrame(table)


def per_class(frame: pd.DataFrame) -> pd.DataFrame:
    accuracy = frame.pivot_table(index=["label_id", "label"], columns="variant",
                                 values="correct", aggfunc="mean")
    cosine = frame.pivot_table(index=["label_id", "label"], columns="variant",
                               values="cosine_to_dense", aggfunc="mean")
    counts = frame[frame["variant"] == "denso"].groupby(["label_id", "label"]).size()
    table = pd.DataFrame({"Clipes": counts})
    for variant in NAMES:
        table[f"Acurácia — {SHORT[variant]}"] = accuracy[variant]
    for variant in list(NAMES)[1:]:
        table[f"Cosseno — {SHORT[variant]}"] = cosine[variant]
    table = table.reset_index().rename(columns={"label_id": "Classe (id)", "label": "Classe"})
    table["Classe (id)"] = table["Classe (id)"].astype("Int64")
    return table


def per_video(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.groupby(["video", "variant"], sort=False).agg(
        Clipes=("correct", "size"), Acuracia=("correct", "mean"),
        Concordancia=("agrees_with_dense", "mean"), Cosseno=("cosine_to_dense", "mean"),
        Tokens=("tokens", "mean"), Tempo=("seconds", "mean"),
    ).reset_index()
    table["variant"] = table["variant"].map(NAMES)
    return table.rename(columns={
        "video": "Vídeo", "variant": "Variante", "Acuracia": "Acurácia top-1",
        "Concordancia": "Concordância com o denso", "Cosseno": "Cosseno médio com o denso",
        "Tokens": "Tokens médios por clipe", "Tempo": "Tempo médio por clipe (s)",
    })


def per_clip(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame["variant"] == "denso"][
        ["video", "start_frame", "start_s", "end_s", "label_id", "label"]
    ].set_index(["video", "start_frame"])
    for variant in NAMES:
        rows = frame[frame["variant"] == variant].set_index(["video", "start_frame"])
        tag = SHORT[variant]
        base[f"{tag}: predição"] = rows["predicted"]
        base[f"{tag}: acerto"] = rows["correct"].map({True: "sim", False: "não"})
        base[f"{tag}: confiança"] = rows["confidence"]
        if variant != "denso":
            base[f"{tag}: cosseno"] = rows["cosine_to_dense"]
            base[f"{tag}: tokens"] = rows["tokens"]
        base[f"{tag}: tempo (s)"] = rows["seconds"]
    table = base.reset_index().sort_values(["video", "start_frame"])
    table["label_id"] = table["label_id"].astype("Int64")
    return table.rename(columns={
        "video": "Vídeo", "start_frame": "Quadro inicial", "start_s": "Início (s)",
        "end_s": "Fim (s)", "label_id": "Classe (id)", "label": "Classe verdadeira",
    })


def configuration(frame: pd.DataFrame) -> pd.DataFrame:
    run = json.loads((RESULTS / "run.json").read_text("utf-8"))
    policy = json.loads(Path(run["policy"]["path"]).read_text("utf-8"))
    region = json.loads(Path(run["region"]["path"]).read_text("utf-8"))
    budgets = frame[frame["variant"] == "vitprune"].groupby("video")["tokens"].value_counts()
    settings = run["settings"]
    device = settings.get("device", "cpu")
    hardware = run["environment"].get("gpu") or "CPU"
    fitted, found = policy["calibration"], region["calibration"]
    items = [
        ("Modelo", "VideoMAEv2-g (ViT-g/14) com a cabeça de 16 classes do ajuste AI City"),
        ("Pesos", run["weights"]["path"]),
        ("Pesos: SHA-256", run["weights"]["sha256"]),
        ("Definição do modelo", run["model_source"]),
        ("Verificação do caminho denso", run["check"]),
        ("Ambiente", f"Python {run['environment']['python']}, torch "
                     f"{run['environment']['torch']}, {hardware} ({device}), "
                     f"precisão {settings.get('precision', 'off')}"),
        ("Política", f"{policy['policy']['version']} calibrada em {fitted['clips']} clipes de "
                     f"{len(fitted['videos'])} vídeos ({run['policy']['path']})"),
        ("Política: níveis e limiares", json.dumps({"levels": policy["policy"]["levels"],
                                                    "thresholds": policy["policy"]["thresholds"]})),
        ("Região do motorista",
         f"{found['area']:.0%} do quadro, motorista detectado em {found['detected']} de "
         f"{found['clips']} clipes de {len(found['videos'])} vídeos ({run['region']['path']})"),
    ]
    items += [(f"Configuração: {k}", json.dumps(v, ensure_ascii=False))
              for k, v in run["settings"].items()]
    for video in videos():
        stem = video.stem
        items += [
            (f"{stem}: clipes avaliados", run["schedules"][stem]),
            (f"{stem}: tokens por clipe (clipes)",
             json.dumps({int(k): int(v) for k, v in budgets.get(stem, pd.Series()).items()})),
        ]
    sampled = settings.get("clips", "sample") == "sample"
    items += [
        ("Nota: ground truth",
         "Acurácia só nos clipes inteiramente dentro de uma ação anotada; "
         + ("4 clipes espaçados por ação." if sampled else
            "todos os clipes do vídeo entram no cosseno, nos tempos e no AMA.")),
        ("Nota: calibração", "A política e a região vêm de vitprune calibrate e vitprune region, "
                             "em vídeos que não são avaliados aqui."),
        ("Nota: pré-processamento", "Quadro inteiro (1920x1080) redimensionado para 224x224, "
                                    "como no pipeline do vitprune. O AMA e o ajuste do "
                                    "VideoMAEv2 usaram recortes do motorista (crop_videos): a "
                                    "acurácia absoluta do denso não é a do AI City; o contraste "
                                    "entre variantes é o que vale."),
        ("Nota: tempos", f"Medidos em {hardware}, {settings.get('batch', 1)} clipe(s) por vez: "
                         "servem para comparar variantes entre si."),
    ]
    return pd.DataFrame(items, columns=["Item", "Valor"])


def ama() -> pd.DataFrame | None:
    """mAP of the AMA localizer per variant, when ``evaluate_ama.py`` has run."""
    path = RESULTS / "ama.json"
    if not path.exists():
        return None
    found = json.loads(path.read_text("utf-8"))
    dense = found.get("denso", {}).get("average")
    table = []
    for variant, result in found.items():
        row = {"Variante": NAMES[variant]}
        row.update({f"mAP@tIoU {tiou}": value for tiou, value in result["map"].items()})
        row["mAP médio"] = result["average"]
        row["Diferença para o denso (p.p.)"] = (
            None if dense is None or result["average"] is None else result["average"] - dense
        )
        table.append(row)
    return pd.DataFrame(table)


def writable(path: Path) -> Path:
    """The path itself, or a timestamped sibling when another program holds it open:
    Windows locks a workbook open in Excel and an image open in some viewers."""
    try:
        path.open("ab").close()
        return path
    except OSError:
        found = path.with_stem(f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}")
        print(f"{path.name} is open elsewhere; writing {found.name} instead")
        return found


def write_excel(sheets: dict[str, pd.DataFrame]) -> None:
    path = writable(RESULTS / "benchmark-vitprune.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for column, title in enumerate(table.columns, start=1):
                width = max(len(str(title)), *(len(f"{v:.3f}" if isinstance(v, float) else str(v))
                                               for v in table[title].head(200)))
                letter = sheet.cell(row=1, column=column).column_letter
                sheet.column_dimensions[letter].width = min(width + 2, 70)
                share = any(k in str(title) for k in ("Acurácia", "Concordância"))
                for row in range(2, len(table) + 2):
                    cell = sheet.cell(row=row, column=column)
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0%" if share else "0.000"
            sheet.freeze_panes = "A2"
    print(f"written {path}")


def cells_to_pixels(cells: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height) * GRID // height
    columns = np.arange(width) * GRID // width
    return cells.reshape(GRID, GRID)[rows[:, None], columns[None, :]]


def shade(image: np.ndarray, inside: np.ndarray, dim: float = 0.3) -> np.ndarray:
    out = image.astype(np.float32)
    out[~cells_to_pixels(inside, *image.shape[:2])] *= dim
    return out.clip(0, 255).astype(np.uint8)


def outline(axis, cells: np.ndarray, height: int, width: int, color: str) -> None:
    for position in np.flatnonzero(cells.reshape(-1)):
        row, column = divmod(int(position), GRID)
        axis.add_patch(Rectangle((column * width / GRID, row * height / GRID), width / GRID,
                                 height / GRID, fill=False, edgecolor=color, linewidth=0.6))


def frame_of(stem: str, start: int) -> np.ndarray:
    image = plt.imread(CACHE / stem / f"{start:06d}.jpg")
    return image[::2, ::2]


def masks_figure(stem: str, start: int, region: np.ndarray, old: np.ndarray) -> None:
    image = frame_of(stem, start)
    height, width = image.shape[:2]
    figure, axes = plt.subplots(1, 2, figsize=(16, 4.9))
    axes[0].imshow(shade(image, old > 0.5, dim=0.35))
    outline(axes[0], old < 0.5, height, width, "orangered")
    axes[0].set_title(f"Máscara antiga: {OLD_MASK['rows']} linhas do topo com peso "
                      f"{OLD_MASK['weight']}".replace(".", ","))
    axes[1].imshow(shade(image, region > 0.5))
    outline(axes[1], region > 0.5, height, width, "magenta")
    axes[1].set_title(f"Região do motorista (YOLO nos vídeos de calibração): "
                      f"{region.mean():.0%} do quadro")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Máscaras aplicadas a {stem}", fontsize=13)
    figure.tight_layout()
    figure.savefig(writable(FIGURES / f"mascaras-{stem}.png"), dpi=110)
    plt.close(figure)


def selection_figure(frame: pd.DataFrame, stem: str, start: int) -> None:
    arrays = np.load(RESULTS / "selection" / f"{stem}_{start:06d}.npz")
    rows = frame[(frame["video"] == stem) & (frame["start_frame"] == start)].set_index("variant")
    if len(rows) < len(NAMES):
        return
    image = frame_of(stem, start)
    height, width = image.shape[:2]
    truth = rows.iloc[0]
    region = arrays["vitprune_regiao_prior"]
    old = arrays["vitprune_mascara_antiga_prior"]

    def verdict(variant: str) -> str:
        row = rows.loc[variant]
        mark = "✓" if row["correct"] else "✗"
        return f"{mark} {row['predicted']} ({row['confidence']:.2f})"

    figure, axes = plt.subplots(2, 3, figsize=(19, 8.4))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Quadro original | denso: {verdict('denso')}", fontsize=10)
    axes[0, 1].imshow(shade(image, old > 0.5, dim=0.35))
    outline(axes[0, 1], old < 0.5, height, width, "orangered")
    axes[0, 1].set_title("Máscara antiga (linhas do topo, peso 0,15)", fontsize=10)
    axes[0, 2].imshow(shade(image, region > 0.5))
    outline(axes[0, 2], region > 0.5, height, width, "magenta")
    axes[0, 2].set_title(f"Região do motorista ({region.mean():.0%} do quadro)", fontsize=10)
    pruned = ("vitprune", "vitprune_mascara_antiga", "vitprune_regiao")
    for axis, variant in zip(axes[1], pruned, strict=True):
        kept = arrays[f"{variant}_kept"][MIDDLE_SLICE]
        axis.imshow(shade(image, kept, dim=0.25))
        outline(axis, kept, height, width, "lime")
        row = rows.loc[variant]
        axis.set_title(f"{SHORT[variant]}: {kept.sum()} de {GRID * GRID} tokens na fatia "
                       f"({int(row['tokens'])} no clipe) | {verdict(variant)}\n"
                       f"cosseno com o denso {row['cosine_to_dense']:.3f}", fontsize=10)
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(f"{stem} | {truth['start_s']:.1f}–{truth['end_s']:.1f} s | "
                    f"classe verdadeira: {truth['label']}  (claro = token mantido, escuro = "
                    f"fundido; fatia temporal central)", fontsize=12)
    figure.tight_layout()
    name = truth["label"].lower().replace(" ", "-").replace("(", "").replace(")", "")
    figure.savefig(writable(FIGURES / f"selecao-{stem[-6:]}-{start:06d}-{name}.png"), dpi=100)
    plt.close(figure)


def summary_figure(frame: pd.DataFrame) -> None:
    """The headline numbers for both videos together and for each one: the budget the
    policy gave each video differs, so the pooled figure alone would mislead."""
    groups = [("Os dois vídeos", frame)]
    for stem in sorted(frame["video"].unique()):
        rows = frame[frame["video"] == stem]
        tokens = rows[rows["variant"] == "vitprune"]["tokens"].mean()
        groups.append((f"Vídeo {stem.rsplit('_', 1)[1]}\n({tokens:.0f} tokens/clipe)", rows))
    colors = {"denso": "#7f7f7f", "vitprune": "#1f77b4",
              "vitprune_mascara_antiga": "#ff7f0e", "vitprune_regiao": "#d62728"}
    metrics = (
        ("Acurácia top-1 (%)", lambda rows, dense: 100 * rows["correct"].mean(), "{:.0f}"),
        ("Cosseno médio com o denso", lambda rows, dense: rows["cosine_to_dense"].mean(),
         "{:.2f}"),
        ("Speedup medido", lambda rows, dense: dense["seconds"].mean() / rows["seconds"].mean(),
         "{:.1f}x"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    width = 0.2
    for axis, (title, measure, label) in zip(axes, metrics, strict=True):
        for k, variant in enumerate(NAMES):
            values = [measure(rows[rows["variant"] == variant], rows[rows["variant"] == "denso"])
                      for _, rows in groups]
            positions = np.arange(len(groups)) + (k - 1.5) * width
            axis.bar(positions, values, width, color=colors[variant], label=SHORT[variant])
            for x, value in zip(positions, values, strict=True):
                axis.text(x, value, label.format(value), ha="center", va="bottom", fontsize=7)
        axis.set_xticks(np.arange(len(groups)), [name for name, _ in groups])
        axis.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(NAMES), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(writable(FIGURES / "resumo.png"), dpi=110)
    plt.close(figure)


def main() -> None:
    frame = load()
    sheets = {
        "Resumo": summary(frame),
        "Por classe": per_class(frame),
        "Por vídeo": per_video(frame),
        "Por clipe": per_clip(frame),
        "Configuração": configuration(frame),
    }
    localized = ama()
    if localized is not None:
        sheets = {"AMA (mAP)": localized, **sheets}
    write_excel(sheets)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary_figure(frame)
    shown = sorted((RESULTS / "selection").glob("*.npz"))
    for path in shown:
        stem, start = path.stem.rsplit("_", 1)
        selection_figure(frame, stem, int(start))
    for video in videos():
        first = next((p for p in shown if p.stem.startswith(video.stem)), None)
        if first is not None:
            arrays = np.load(first)
            masks_figure(video.stem, int(first.stem.rsplit("_", 1)[1]),
                         arrays["vitprune_regiao_prior"], arrays["vitprune_mascara_antiga_prior"])
    print(f"{len(frame) // len(NAMES)} clips, figures in {FIGURES}")


if __name__ == "__main__":
    main()
