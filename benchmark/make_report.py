"""Turn the benchmark records into a spreadsheet and figures.

Reads what ``run_inference.py`` and, once it has run, ``evaluate_ama.py`` wrote under
``results``, and writes:

* ``results/benchmark-vitprune.xlsx`` -- the summary per variant; every annotated action
  read whole, per variant: the class the fine-tuned head gives it from all its clips and
  the AMA detection that matches it; the tokens kept per video; the AMA's mAP; and the
  configuration the numbers were produced under;
* ``results/figuras/`` -- the fixed and the smart mask over a frame of each video, the
  tokens each variant kept in the middle clip of each action, the AMA's detections along
  each video and a chart of the summary.

Results are read per action, never per clip: a clip covers about a second, while the
pipeline answers about whole actions.
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
from common import CACHE, GEOMETRY, RESULTS, VARIANTS, annotations, segments, videos  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from run_inference import FIXED_MASK, PRUNE_LAYER  # noqa: E402

from vitprune.bench.cost import budget_cost  # noqa: E402

NAMES = {
    "denso": "Sem ViT-Prune (denso)",
    "vitprune": "ViT-Prune sem máscara",
    "vitprune_mascara_fixa": "ViT-Prune + máscara fixa",
    "vitprune_mascara_inteligente": "ViT-Prune + máscara inteligente",
}
SHORT = {
    "denso": "Denso",
    "vitprune": "Sem máscara",
    "vitprune_mascara_fixa": "Máscara fixa",
    "vitprune_mascara_inteligente": "Máscara inteligente",
}
COLORS = {"denso": "#7f7f7f", "vitprune": "#1f77b4", "vitprune_mascara_fixa": "#ff7f0e",
          "vitprune_mascara_inteligente": "#d62728"}
PRUNED = VARIANTS[1:]
FIGURES = RESULTS / "figuras"
GRID = GEOMETRY.grid_size
MIDDLE_SLICE = GEOMETRY.temporal_slices // 2
SHOWN_SCORE = 0.3
#: Columns shown as percentages in the workbook.
PERCENT = ("Ações", "(%)", "região")


def load() -> pd.DataFrame:
    rows = [json.loads(line) for line in (RESULTS / "clips.jsonl").read_text("utf-8").splitlines()]
    return pd.DataFrame(rows)


def recorded() -> tuple[dict, dict, dict]:
    """``run.json``, and the policy and region files it names."""
    run = json.loads((RESULTS / "run.json").read_text("utf-8"))
    policy = json.loads(Path(run["policy"]["path"]).read_text("utf-8"))
    region = json.loads(Path(run["region"]["path"]).read_text("utf-8"))
    return run, policy, region


def class_names() -> dict[int, str]:
    return {a["label_id"]: a["label"]
            for video in annotations()["database"].values() for a in video["annotations"]}


def ama_results() -> dict | None:
    """mAP of the AMA per variant, when ``evaluate_ama.py`` has run."""
    path = RESULTS / "ama.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def ama_detections() -> dict[str, pd.DataFrame] | None:
    """The AMA's detections per variant, in seconds, when ``evaluate_ama.py`` has run."""
    paths = {variant: RESULTS / "ama" / variant / "predictions.csv" for variant in VARIANTS}
    if not all(path.exists() for path in paths.values()):
        return None
    ids = sorted({s["label_id"] for video in videos() for s in segments(video.stem)})
    # The AMA evaluator renumbers the classes by the label ids annotated in the evaluated
    # videos, which leaves them unchanged only when every class is annotated there.
    if ids != list(range(len(ids))):
        raise ValueError(f"the annotations hold classes {ids}; the AMA's labels cannot be read")
    return {variant: pd.read_csv(path) for variant, path in paths.items()}


def match(detections: pd.DataFrame, stem: str, action: dict) -> tuple[float, float] | None:
    """The most confident detection of the action's class that overlaps it: its tIoU
    and its confidence."""
    same = detections[(detections["video-id"] == stem)
                      & (detections["label"] == action["label_id"])]
    inter = (np.minimum(same["t-end"], action["end"])
             - np.maximum(same["t-start"], action["start"])).clip(lower=0)
    union = np.maximum(same["t-end"], action["end"]) - np.minimum(same["t-start"], action["start"])
    overlapping = same[inter > 0]
    if overlapping.empty:
        return None
    best = overlapping["score"].idxmax()
    return float(inter[best] / union[best]), float(overlapping.loc[best, "score"])


def actions(frame: pd.DataFrame, detections: dict | None) -> pd.DataFrame:
    """Every annotated action read whole. The class in the stretch averages the head's
    probabilities over the clips wholly inside the action."""
    names = class_names()
    table = []
    for video in videos():
        stem = video.stem
        rows = frame[frame["video"] == stem]
        probabilities = {}
        for variant in VARIANTS:
            mine = rows[rows["variant"] == variant].sort_values("start_frame")
            found = np.load(RESULTS / "features" / variant / f"{stem}.npz")["a_prob"]
            if len(found) != len(mine):
                raise ValueError(f"{variant}/{stem}.npz holds {len(found)} clips, "
                                 f"clips.jsonl {len(mine)}; the run is incomplete")
            probabilities[variant] = (mine["segment"].to_numpy(), found)
        for index, action in enumerate(segments(stem)):
            inside = rows[(rows["segment"] == index) & (rows["variant"] == "vitprune")]
            row = {"Vídeo": stem, "Ação": action["label"], "Início (s)": action["start"],
                   "Fim (s)": action["end"], "Clipes no trecho": len(inside),
                   "Tokens médios (ViT-Prune)": inside["tokens"].mean()}
            for variant in VARIANTS:
                tag, (owners, found) = SHORT[variant], probabilities[variant]
                picked = found[owners == index]
                predicted = int(picked.mean(axis=0).argmax()) if len(picked) else None
                row[f"{tag}: classe no trecho"] = names.get(predicted, predicted)
                row[f"{tag}: acerto"] = (None if predicted is None
                                         else predicted == action["label_id"])
                if detections is not None:
                    matched = match(detections[variant], stem, action)
                    row[f"{tag}: AMA tIoU"] = 0.0 if matched is None else matched[0]
                    row[f"{tag}: AMA confiança"] = None if matched is None else matched[1]
            table.append(row)
    return pd.DataFrame(table)


def summary(frame: pd.DataFrame, by_action: pd.DataFrame, ama: dict | None,
            area: float) -> pd.DataFrame:
    dense_seconds = frame[frame["variant"] == "denso"]["seconds"].mean()
    dense_classes = by_action["Denso: classe no trecho"]
    table = []
    for variant in VARIANTS:
        rows, tag = frame[frame["variant"] == variant], SHORT[variant]
        tokens = rows["tokens"].astype(int).tolist()
        layer = 0 if variant == "denso" else PRUNE_LAYER
        judged = by_action[f"{tag}: acerto"].notna()
        row = {"Variante": NAMES[variant]}
        if ama is not None:
            average, dense = ama[variant]["average"], ama["denso"]["average"]
            row["mAP médio do AMA"] = average
            row["Diferença de mAP para o denso (p.p.)"] = (
                None if average is None or dense is None else average - dense)
        row |= {
            "Ações reconhecidas no trecho": by_action.loc[judged, f"{tag}: acerto"]
            .astype(float).mean(),
            "Ações com a classe do denso": (by_action[f"{tag}: classe no trecho"]
                                            == dense_classes)[judged].mean(),
            "Tokens médios por clipe": np.mean(tokens),
            "Tokens mantidos (%)": np.mean(tokens) / GEOMETRY.num_tokens,
            # Dense keeps every token, so its share is the region's area.
            "Tokens na região do motorista": area if variant == "denso"
            else rows["inside_region"].mean(),
            "Cosseno médio com o denso": rows["cosine_to_dense"].mean(),
            "Speedup previsto": budget_cost(GEOMETRY, tokens, layer).predicted_speedup,
            "Speedup medido": dense_seconds / rows["seconds"].mean(),
            "Tempo médio por clipe (s)": rows["seconds"].mean(),
        }
        table.append(row)
    return pd.DataFrame(table)


def per_video(frame: pd.DataFrame, area: float, levels: list[int]) -> pd.DataFrame:
    table = []
    for (video, variant), rows in frame.groupby(["video", "variant"], sort=False):
        row = {"Vídeo": video, "Variante": NAMES[variant], "Clipes": len(rows),
               "Tokens médios por clipe": rows["tokens"].mean()}
        for level in levels:
            row[f"Clipes com {level} tokens (%)"] = (
                None if variant == "denso" else (rows["tokens"] == level).mean())
        row["Tokens na região do motorista"] = (
            area if variant == "denso" else rows["inside_region"].mean())
        row["Cosseno médio com o denso"] = rows["cosine_to_dense"].mean()
        row["Tempo médio por clipe (s)"] = rows["seconds"].mean()
        table.append(row)
    return pd.DataFrame(table)


def ama_table(ama: dict) -> pd.DataFrame:
    dense = ama["denso"]["average"]
    table = []
    for variant, result in ama.items():
        row = {"Variante": NAMES[variant]}
        row.update({f"mAP@tIoU {tiou}": value for tiou, value in result["map"].items()})
        row["mAP médio"] = result["average"]
        row["Diferença para o denso (p.p.)"] = (
            None if dense is None or result["average"] is None else result["average"] - dense
        )
        table.append(row)
    return pd.DataFrame(table)


def configuration(run: dict, policy: dict, region: dict) -> pd.DataFrame:
    settings = run["settings"]
    hardware = run["environment"].get("gpu") or "CPU"
    fitted, found = policy["calibration"], region["calibration"]
    detector = found.get("detector") or {}
    detector_weights = Path(detector.get("weights", {}).get("path", "?")).name
    items = [
        ("Modelo", "VideoMAEv2-g (ViT-g/14) com a cabeça de 16 classes do ajuste AI City"),
        ("Pesos", run["weights"]["path"]),
        ("Pesos: SHA-256", run["weights"]["sha256"]),
        ("Definição do modelo", run["model_source"]),
        ("Verificação do caminho denso", run["check"]),
        ("Ambiente", f"Python {run['environment']['python']}, torch "
                     f"{run['environment']['torch']}, {hardware} ({settings['device']}), "
                     f"precisão {settings['precision']}"),
        ("Política", f"{policy['policy']['version']} calibrada em {fitted['clips']} clipes de "
                     f"{len(fitted['videos'])} vídeos ({run['policy']['path']})"),
        ("Política: níveis e limiares", json.dumps({"levels": policy["policy"]["levels"],
                                                    "thresholds": policy["policy"]["thresholds"]})),
        ("Máscara fixa", f"{FIXED_MASK['rows']} linhas do topo com peso "
                         f"{FIXED_MASK['weight']}".replace(".", ",")),
        ("Máscara inteligente",
         f"região do motorista: {found['area']:.0%} do quadro, motorista detectado em "
         f"{found['detected']} de {found['clips']} clipes de {len(found['videos'])} vídeos, "
         f"detector {detector_weights} (Ultralytics {detector.get('version', '?')}) "
         f"({run['region']['path']})"),
    ]
    items += [(f"Configuração: {k}", json.dumps(v, ensure_ascii=False))
              for k, v in settings.items()]
    items += [(f"{stem}: clipes", count) for stem, count in run["schedules"].items()]
    items += [
        ("Nota: leitura por ação",
         "Cada ação anotada é lida inteira. Classe no trecho: média das probabilidades da "
         "cabeça nos clipes inteiramente dentro da ação. AMA: a detecção mais confiante da "
         "classe da ação que se sobrepõe a ela, com o seu tIoU."),
        ("Nota: máscaras", "As três variantes do ViT-Prune usam a mesma política e mantêm os "
                           "mesmos tokens por clipe; as máscaras só mudam quais tokens ficam."),
        ("Nota: calibração", "A política e a região vêm de vitprune calibrate e vitprune region, "
                             "em vídeos que não são avaliados aqui."),
        ("Nota: pré-processamento", "Quadro inteiro (1920x1080) redimensionado para 224x224, "
                                    "como no pipeline do vitprune. O AMA e o ajuste do "
                                    "VideoMAEv2 usaram recortes do motorista (crop_videos): a "
                                    "acurácia absoluta do denso não é a do AI City; o contraste "
                                    "entre variantes é o que vale."),
        ("Nota: tempos", f"Medidos em {hardware}, {settings['batch']} clipe(s) por vez: "
                         "servem para comparar variantes entre si."),
    ]
    return pd.DataFrame(items, columns=["Item", "Valor"])


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
                share = any(k in str(title) for k in PERCENT)
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


def show_fixed(axis, image: np.ndarray, fixed: np.ndarray) -> None:
    height, width = image.shape[:2]
    axis.imshow(shade(image, fixed > 0.5, dim=0.35))
    outline(axis, fixed < 0.5, height, width, "orangered")


def show_smart(axis, image: np.ndarray, smart: np.ndarray) -> None:
    height, width = image.shape[:2]
    axis.imshow(shade(image, smart > 0.5))
    outline(axis, smart > 0.5, height, width, "magenta")


def masks_figure(stem: str, start: int, smart: np.ndarray, fixed: np.ndarray) -> None:
    image = frame_of(stem, start)
    figure, axes = plt.subplots(1, 2, figsize=(16, 4.9))
    show_fixed(axes[0], image, fixed)
    axes[0].set_title(f"Máscara fixa: {FIXED_MASK['rows']} linhas do topo com peso "
                      f"{FIXED_MASK['weight']}".replace(".", ","))
    show_smart(axes[1], image, smart)
    axes[1].set_title(f"Máscara inteligente (YOLO nos vídeos de calibração): "
                      f"{smart.mean():.0%} do quadro")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Máscaras aplicadas a {stem}", fontsize=13)
    figure.tight_layout()
    figure.savefig(writable(FIGURES / f"mascaras-{stem}.png"), dpi=110)
    plt.close(figure)


def selection_figure(frame: pd.DataFrame, stem: str, start: int) -> None:
    arrays = np.load(RESULTS / "selection" / f"{stem}_{start:06d}.npz")
    rows = frame[(frame["video"] == stem) & (frame["start_frame"] == start)].set_index("variant")
    image = frame_of(stem, start)
    height, width = image.shape[:2]
    action = segments(stem)[int(rows["segment"].iloc[0])]

    figure, axes = plt.subplots(2, 3, figsize=(19, 8.4))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Quadro original", fontsize=10)
    show_fixed(axes[0, 1], image, arrays["vitprune_mascara_fixa_prior"])
    axes[0, 1].set_title("Máscara fixa (linhas do topo, peso 0,15)", fontsize=10)
    smart = arrays["vitprune_mascara_inteligente_prior"]
    show_smart(axes[0, 2], image, smart)
    axes[0, 2].set_title(f"Máscara inteligente ({smart.mean():.0%} do quadro)", fontsize=10)
    for axis, variant in zip(axes[1], PRUNED, strict=True):
        kept = arrays[f"{variant}_kept"][MIDDLE_SLICE]
        axis.imshow(shade(image, kept, dim=0.25))
        outline(axis, kept, height, width, "lime")
        row = rows.loc[variant]
        axis.set_title(f"{SHORT[variant]}: {kept.sum()} de {GRID * GRID} tokens na fatia "
                       f"({int(row['tokens'])} no clipe)\n"
                       f"{row['inside_region']:.0%} dos tokens na região do motorista",
                       fontsize=10)
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(f"{stem} | {action['label']} ({action['start']:.1f}–{action['end']:.1f} s), "
                    f"clipe em {rows['start_s'].iloc[0]:.1f} s  (claro = token mantido, escuro = "
                    f"fundido; fatia temporal central)", fontsize=12)
    figure.tight_layout()
    name = action["label"].lower().replace(" ", "-").replace("(", "").replace(")", "")
    figure.savefig(writable(FIGURES / f"selecao-{stem[-6:]}-{start:06d}-{name}.png"), dpi=100)
    plt.close(figure)


def timeline_figure(stem: str, detections: dict[str, pd.DataFrame]) -> None:
    """What the AMA found along the video, per variant, against the annotation."""
    names = class_names()
    colors = dict(zip(sorted(names), plt.cm.tab20.colors, strict=False))
    lanes = [("Anotação real", [(a["start"], a["end"], a["label_id"], 1.0)
                                for a in segments(stem)])]
    for variant in VARIANTS:
        found = detections[variant]
        found = found[(found["video-id"] == stem) & (found["score"] >= SHOWN_SCORE)]
        lanes.append((f"AMA: {SHORT[variant]}",
                      list(zip(found["t-start"], found["t-end"], found["label"], found["score"],
                               strict=True))))
    figure, axis = plt.subplots(figsize=(22, 4.8))
    for lane, (_, bars) in enumerate(lanes):
        for start, end, label, score in sorted(bars, key=lambda bar: bar[3]):
            axis.barh(lane, end - start, left=start, height=0.7, color=colors[label],
                      alpha=0.35 + 0.65 * score)
    axis.set_yticks(range(len(lanes)), [title for title, _ in lanes])
    axis.invert_yaxis()
    axis.set_xlabel("segundos no vídeo")
    axis.set_title(f"{stem}: ações encontradas pelo AMA (confiança ≥ {SHOWN_SCORE}; "
                   "mais opaco = mais confiante)")
    figure.legend(handles=[Patch(color=colors[k], label=names[k]) for k in sorted(names)],
                  loc="lower center", ncol=8, fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(writable(FIGURES / f"linha-do-tempo-{stem}.png"), dpi=110)
    plt.close(figure)


def summary_figure(table: pd.DataFrame) -> None:
    metrics = [
        ("mAP médio do AMA", "mAP médio do AMA", 1, "{:.1f}"),
        ("Ações reconhecidas no trecho", "Ações reconhecidas no trecho (%)", 100, "{:.0f}"),
        ("Tokens na região do motorista", "Tokens na região do motorista (%)", 100, "{:.0f}"),
        ("Tokens médios por clipe", "Tokens médios por clipe", 1, "{:.0f}"),
        ("Speedup medido", "Speedup medido", 1, "{:.1f}x"),
    ]
    metrics = [m for m in metrics if m[0] in table and table[m[0]].notna().any()]
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.6))
    for axis, (column, title, scale, label) in zip(np.atleast_1d(axes), metrics, strict=True):
        values = (table[column] * scale).tolist()
        axis.bar(range(len(VARIANTS)), values, color=[COLORS[v] for v in VARIANTS])
        for x, value in enumerate(values):
            if value == value:  # skips NaN
                axis.text(x, value, label.format(value), ha="center", va="bottom", fontsize=8)
        axis.set_xticks(range(len(VARIANTS)), [SHORT[v] for v in VARIANTS], rotation=20,
                        fontsize=8)
        axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(writable(FIGURES / "resumo.png"), dpi=110)
    plt.close(figure)


def main() -> None:
    frame = load()
    run, policy, region = recorded()
    area = float(np.mean(region["region"]["prior"]))
    ama, detections = ama_results(), ama_detections()
    by_action = actions(frame, detections)
    table = summary(frame, by_action, ama, area)
    sheets = {"Resumo": table, "Por ação": by_action,
              "Por vídeo": per_video(frame, area, policy["policy"]["levels"])}
    if ama is not None:
        sheets["AMA (mAP)"] = ama_table(ama)
    sheets["Configuração"] = configuration(run, policy, region)
    write_excel(sheets)

    FIGURES.mkdir(parents=True, exist_ok=True)
    summary_figure(table)
    shown = sorted((RESULTS / "selection").glob("*.npz"))
    for path in shown:
        stem, start = path.stem.rsplit("_", 1)
        selection_figure(frame, stem, int(start))
    for video in videos():
        first = next((p for p in shown if p.stem.rsplit("_", 1)[0] == video.stem), None)
        if first is not None:
            arrays = np.load(first)
            masks_figure(video.stem, int(first.stem.rsplit("_", 1)[1]),
                         arrays["vitprune_mascara_inteligente_prior"],
                         arrays["vitprune_mascara_fixa_prior"])
        if detections is not None:
            timeline_figure(video.stem, detections)
    print(f"{len(by_action)} actions, figures in {FIGURES}")


if __name__ == "__main__":
    main()
