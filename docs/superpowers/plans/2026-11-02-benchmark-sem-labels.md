# Benchmark sem labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vitprune benchmark run | ama | report` funcionam sem arquivo de anotação, comparando cada variante com o pipeline denso na cabeça de classificação e nas detecções do AMA.

**Architecture:** `run` deixa de atribuir clipes a ações anotadas e registra fps/quadros/clipes por vídeo no `run.json`. `ama` roda o modelo do AMA direto sobre as features, num script autocontido lançado como subprocesso no interpretador do AMA, sem `eval.py`. `report` usa o denso como referência (concordância de classe, recall, precisão e tIoU médio).

**Tech Stack:** Python 3.10, torch, numpy, pandas, matplotlib, openpyxl, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-02-benchmark-sem-labels-design.md`

**Sem commits neste plano.** A árvore ainda tem a migração do benchmark para o CLI sem commitar; quando e como commitar fica com o dono do repositório.

**Ordem.** Tasks 1 a 3 compartilham o `RunRecord`. Ao fim da Task 1 os comandos `ama` e `report` ficam quebrados (usavam as anotações) e os testes deles são apagados; as Tasks 2 e 3 os refazem. Nada é entregue no meio.

## Global Constraints

- O AMA fica como está, rodando no interpretador dele (loguru, torch), com o `nms_1d_cpu` em Python no lugar da extensão compilada.
- O formato das features (`feats`, `a_prob`) não muda.
- Uma variante já avaliada pelo AMA é recusada, como hoje.
- O relatório diz "concordância com o denso", nunca "acerto".
- Padrões: `--min-score` 0,3, `--tiou` 0,5, `--show` 6.
- Saída do AMA: `ama/<variante>/predictions.csv` com as colunas `video-id, t-start, t-end, label, score`.
- `ama_infer.py` não importa o vitprune.
- Nenhuma dependência nova. Python ≥ 3.10, ruff com linha de 100: rodar `ruff check src tests` (nunca `.`, que varre `benchmark/.deps`).
- `cli.py` só traduz argumentos; `main` transforma `OSError`, `ValueError` e `KeyError` em `error: ...` (código 2).

## Review Focus

Cada linha tem o teste na task dona.

1. Uma variante sem nenhuma detecção: o AMA grava só o cabeçalho, recall 0, precisão indefinida, sem quebrar (Tasks 2 e 3).
2. O denso sem detecção acima do limiar: recall e tIoU indefinidos, o relatório sai (Task 3).
3. Um `run.json` de versão anterior (clipes como inteiro, não por vídeo): `ValueError` mandando rodar de novo (Task 1).
4. Saída do AMA já existente em qualquer variante: recusa antes de iniciar o subprocesso e sem criar nada (Task 2). Pesos do AMA que não casam com o modelo: falha sem gravar `predictions.csv` (Task 2).
5. `--show` maior que o número de clipes mostra todos; `--show 0` é recusado (Task 1).

---

### Task 1: `run` sem anotações

**Files:**
- Delete: `src/vitprune/pipeline/benchmark/annotations.py`, `tests/benchmark/test_report.py` (recriado na Task 3)
- Modify: `src/vitprune/pipeline/benchmark/settings.py`, `src/vitprune/pipeline/benchmark/inference.py`, `src/vitprune/pipeline/cli.py` (handler e parser de `benchmark run`), `tests/benchmark/test_run.py`, `tests/benchmark/test_benchmark.py`

**Interfaces:**
- Produces, em `settings.py`: `Benchmark` sem o campo `annotations` e com `show: int = 6` (`< 1` levanta `ValueError`). `RunRecord(out: Path, run: dict)`, só esses dois campos, com `RunRecord.read(out) -> RunRecord` (`FileNotFoundError` mandando rodar `vitprune benchmark run` se não há `run.json`; `ValueError` com "run again" se `schedules` não mapeia cada vídeo para `{"fps", "frames", "clips"}`), `stems`, `settings`, `fixed_mask`, `prune_layer`, `frames`, `figures` como hoje, e `video(stem) -> dict` com esse trio. Saem `database`, `segments` e `class_names`.
- Produces, em `inference.py`: `video_info(video: Path) -> dict` (`{"fps": float, "frames": int}`) e `plan(bench: Benchmark, video: Path, info: dict) -> list[dict]`, com clipes `{"video", "start", "fps", "show"}`. Exatamente `min(bench.show, n)` clipes têm `show=True`, espalhados por igual com o primeiro e o último incluídos (use `vitprune.pipeline.video._spread`).
- Produces, em disco: `run.json` com `"schedules": {stem: {"fps", "frames", "clips"}}`, `settings["show"]` e sem a chave `annotations`; as linhas do `clips.jsonl` sem `segment`.
- Consumes: `Benchmark.clip_starts`, `Benchmark.evaluated` (inalterados).

- [ ] **Step 1: Atualizar os testes**

`tests/benchmark/test_run.py`: o fixture `bench` perde o `class-intervals.json` e o argumento `annotations`, e passa `show=2`. Apagar `test_clips_belong_to_the_action_that_holds_them_whole` e adaptar:
```python
def test_the_run_writes_what_the_next_steps_read(bench, model, geometry):
    inference.run_benchmark(bench)
    clips = len(bench.clip_starts(FPS * SECONDS))
    record = json.loads(bench.record.read_text("utf-8"))
    assert record["schedules"] == {"v": {"fps": float(FPS), "frames": FPS * SECONDS, "clips": clips}}
    assert "annotations" not in record and record["settings"]["show"] == 2
    assert all("segment" not in row for row in _rows(bench))
    # restante do teste atual: linhas por variante, features, manifestos

def test_the_figures_show_clips_spread_over_the_video(bench, model):
    bench.show = 3
    inference.run_benchmark(bench)
    starts = sorted(int(p.stem.rsplit("_", 1)[1]) for p in bench.selections.glob("v_*.npz"))
    assert starts == [0, 32, 48]  # 4 clips (0, 16, 32, 48); first and last included
    assert len(list((bench.frames / "v").glob("*.jpg"))) == 3

def test_show_larger_than_the_clips_shows_them_all(bench, model):
    bench.show = 99
    inference.run_benchmark(bench)
    assert len(list(bench.selections.glob("v_*.npz"))) == 4
```
`test_the_inputs_of_the_figures_are_kept` passa a esperar 2 clipes (o `show=2` do fixture).

`tests/benchmark/test_benchmark.py`: `_benchmark` sem `annotations`; apagar `_annotations`, `test_a_clip_belongs_to_the_action_that_holds_it_whole`, `test_a_video_without_annotation_is_named`, os testes de `ama.labels` e `ama.compat_on_path`, os três testes de `RunRecord` com anotações, `test_the_ama_step_takes_the_model_from_its_flags` e `test_the_run_defaults_to_the_annotations_next_to_the_videos`; tirar `--annotations` de `test_every_setting_of_the_run_comes_from_its_flags` e acrescentar `--show 3` com `assert bench.show == 3`. Acrescentar `{"show": 0}` à parametrização dos ajustes inválidos e:
```python
def test_a_run_of_an_older_version_is_refused(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"schedules": {"v": 3}}), encoding="utf-8")
    with pytest.raises(ValueError, match="run again"):
        RunRecord.read(tmp_path)

def test_a_record_lists_the_evaluated_videos(tmp_path):
    video = {"fps": 8.0, "frames": 64, "clips": 4}
    (tmp_path / "run.json").write_text(json.dumps({"schedules": {"v": video}}), encoding="utf-8")
    record = RunRecord.read(tmp_path)
    assert record.stems == ["v"] and record.video("v")["frames"] == 64
```
Apagar `tests/benchmark/test_report.py`.

- [ ] **Step 2: Rodar e ver falhar**

Run: `PYTHONPATH=src python3 -m pytest tests/benchmark -o addopts="" -q`
Expected: falhas de `annotations`/`show` inexistentes.

- [ ] **Step 3: Implementar** as interfaces acima. `plan` calcula os inícios com `bench.clip_starts(info["frames"])`. `run_benchmark` não lê anotações, monta `schedules` com `video_info` e o número de clipes, e registra `show` em `settings`. Em `cli.py`, `benchmark run` perde `--annotations` e ganha `--show` (padrão 6). Remover o import de `annotations` e apagar `annotations.py`.

- [ ] **Step 4: Rodar e ver passar**

Run: `PYTHONPATH=src python3 -m pytest tests/benchmark tests/pipeline -o addopts="" -q` e `ruff check src tests`
Expected: tudo passa.

---

### Task 2: o AMA direto sobre as features

**Files:**
- Create: `src/vitprune/pipeline/benchmark/ama_infer.py`, `tests/benchmark/test_ama.py`
- Modify: `src/vitprune/pipeline/benchmark/ama.py` (reescrito), `src/vitprune/pipeline/cli.py` (`benchmark ama`)

**Interfaces:**
- Consumes: `RunRecord.read(out)`, `record.stems`, `record.settings["stride"]`, `record.video(stem)` da Task 1; `Manifest`, `SourceFile` de `vitprune.pipeline.provenance`.
- Produces: `run_ama(out: str | Path, project: str | Path, config: str | Path, weights: str | Path, python: str | None = None, progress=print) -> Path`, que devolve `out/ama`.
  - Antes de iniciar: `FileNotFoundError` se faltam `project/libs`, a configuração, os pesos ou algum `features/<variante>/<video>.npz`; `FileExistsError` se qualquer `ama/<variante>/predictions.csv` já existe, sem criar nada.
  - Escreve `ama/job.json` com `{"stride", "variants": {nome: {"features", "destination"}}, "videos": {stem: {"fps", "frames"}}}` e lança `[python or sys.executable, ama_infer.py, config, weights, job.json]` com `cwd=project` e `PYTHONPATH = [COMPAT, project, PYTHONPATH]`. A saída vai para `progress` e para `ama/downstream.log`; código ≠ 0 levanta `RuntimeError("the AMA exited with code N; its output is in <log>")`.
  - Depois, um `ama-manifest.json` por variante: `Manifest(settings={"stage": "ama", "variant", "features", "stride", "config": SourceFile}, weights=SourceFile.of(weights), statistics={"detections": n})`.
- Produces, em `ama_infer.py`: `main(argv: list[str]) -> int`, autocontido (stdlib, numpy, torch e as bibliotecas do AMA). Carrega a configuração (`libs.core.load_config`) e o modelo (`libs.modeling.make_meta_arch(cfg["model_name"], **cfg["model"])`) uma vez, no dispositivo `cfg["devices"][0]`; carrega `state_dict_ema` sem o prefixo `module.` e recusa (`RuntimeError`) se sobrar ou faltar chave. Semente 0 (`libs.utils.fix_random_seed`), `model.eval()`, `torch.no_grad()`. Por vídeo: entrada `{"video_id", "feats": tensor [C, T], "segments": None, "labels": None, "fps", "duration", "feat_stride", "feat_num_frames", "segmentation_labels": None}`, com `fps = cfg["dataset"]["default_fps"] or fps do vídeo`, `duration = frames / fps`, `feat_stride = stride do job`, `feat_num_frames = cfg["dataset"]["num_frames"]`; chama `model([item], is_training=False)`. Cada variante só grava seu `predictions.csv` (colunas em Global Constraints, sem índice) depois de todos os vídeos.
- O CLI `benchmark ama` passa a ter `--out`, `--ama-project`, `--ama-config`, `--ama-weights` e `--ama-python`. Saem `--entry-point` e `--annotations`.

- [ ] **Step 1: Escrever os testes** em `tests/benchmark/test_ama.py`. Um fixture `fake_ama` cria em `tmp_path` um projeto falso: `libs/core/__init__.py` (`load_config` lê o YAML e define `model_name`), `libs/modeling/__init__.py` (`make_meta_arch(name, empty=False)` devolve um `nn.Module` com `nn.Linear(4, 4)`; o `forward(video_list, is_training=False)` devolve, por item, `{"video_id", "segments": tensor([[T, duration]]), "scores": tensor([0.5]), "labels": tensor([feat_stride])}` com `T = feats.shape[-1]`, ou tensores vazios se `empty`), `libs/utils/__init__.py` (`fix_random_seed`), uma configuração YAML (`devices: ['cpu']`, `dataset: {feat_stride: 99, num_frames: 16, default_fps: 30}`), pesos `{"state_dict_ema": {"module." + k: v}}` e uma pasta `out` com `run.json` (`schedules: {"v": {"fps": 8.0, "frames": 64, "clips": 4}}`, `settings: {"stride": 16, "frame_step": 2}`) e `features/<variante>/v.npz` com `feats` zeros `(4, 4)` nas quatro variantes.
```python
def test_the_ama_runs_every_variant_without_annotations(fake_ama):
    run_ama(**fake_ama.args)
    for variant in VARIANTS:
        rows = list(csv.reader((fake_ama.out / "ama" / variant / "predictions.csv").open()))
        assert rows[0] == ["video-id", "t-start", "t-end", "label", "score"]
        video, start, end, label, score = rows[1]
        assert (video, float(start), label) == ("v", 4.0, "16")  # T, and the run's stride, not the config's 99
        assert float(end) == pytest.approx(64 / 30)  # frames over the AMA's default fps
        assert (fake_ama.out / "ama" / variant / "ama-manifest.json").exists()
    assert (fake_ama.out / "ama" / "downstream.log").exists()

def test_a_variant_without_detections_gets_only_the_header(fake_ama):
    fake_ama.set_model_option(empty=True)
    run_ama(**fake_ama.args)
    assert len((fake_ama.out / "ama" / "denso" / "predictions.csv").read_text().splitlines()) == 1

def test_an_existing_output_is_refused_before_anything_runs(fake_ama):
    (fake_ama.out / "ama" / "vitprune").mkdir(parents=True)
    (fake_ama.out / "ama" / "vitprune" / "predictions.csv").write_text("x")
    with pytest.raises(FileExistsError):
        run_ama(**fake_ama.args)
    assert not (fake_ama.out / "ama" / "denso").exists()

def test_weights_that_do_not_match_the_model_write_nothing(fake_ama):
    torch.save({"state_dict_ema": {"other": torch.zeros(1)}}, fake_ama.weights)
    with pytest.raises(RuntimeError, match="exited with code"):
        run_ama(**fake_ama.args)
    assert not list((fake_ama.out / "ama").glob("*/predictions.csv"))

def test_the_inference_script_stays_independent_of_vitprune():
    tree = ast.parse(Path(ama_infer.__file__).read_text("utf-8"))
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(name.split(".")[0] == "vitprune" for name in names)

def test_the_ama_step_takes_everything_from_its_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ama, "run_ama", lambda out, project, config, weights, python=None: seen.update(
        out=out, project=project, config=config, weights=weights, python=python))
    command = ["benchmark", "ama", "--out", "o", "--ama-project", "p", "--ama-config", "c.yaml",
               "--ama-weights", "a.pth", "--ama-python", "/x/python"]
    assert main(command) == 0
    assert seen == {"out": "o", "project": "p", "config": "c.yaml", "weights": "a.pth",
                    "python": "/x/python"}
```
O fixture expõe `args` (dicionário para `run_ama`, com `python=sys.executable`), `out`, `weights` e `set_model_option`, que reescreve a seção `model:` do YAML.

- [ ] **Step 2: Rodar e ver falhar**

Run: `PYTHONPATH=src python3 -m pytest tests/benchmark/test_ama.py -o addopts="" -q`
Expected: erro de importação de `ama_infer` / assinatura de `run_ama`.

- [ ] **Step 3: Implementar** `ama_infer.py` e `ama.py` como descrito, apagando `labels`, `compat_on_path`, `PER_TIOU`, `AVERAGE` e o uso de `DownstreamModel`/`evaluate`. `COMPAT` continua apontando para `ama_compat/`. Atualizar o handler e o parser do CLI.

- [ ] **Step 4: Rodar e ver passar**

Run: `PYTHONPATH=src python3 -m pytest tests/benchmark tests/pipeline -o addopts="" -q` e `ruff check src tests`
Expected: tudo passa.

- [ ] **Step 5: Conferir com o AMA real, à mão.** Montar uma pasta de execução sintética em `/tmp/opencode/run_ama/` (um `run.json` mínimo como o do fixture e `features/<variante>/v.npz` com `feats` aleatório `(40, 1408)`) e rodar:
`PYTHONPATH=src python3 -m vitprune.pipeline.cli benchmark ama --out /tmp/opencode/run_ama --ama-project /home/gsilv257/DAR-sys/models-weights/AIcity2024-track3/AMA --ama-config .../configs/aicity_ego_vitl.yaml --ama-weights /home/gsilv257/DAR-sys/models-weights/AMA/ckpt/aicity_ego_vitl_ckpt/best_performance.pth.tar`
Expected: quatro `predictions.csv` com cabeçalho e linhas de score entre 0 e 1 e label entre 0 e 15; uma segunda execução é recusada.

---

### Task 3: `report` com o denso como referência

**Files:**
- Create: `tests/benchmark/test_report.py`
- Modify: `src/vitprune/pipeline/benchmark/report.py`, `src/vitprune/pipeline/cli.py` (`benchmark report`)

**Interfaces:**
- Consumes: `RunRecord` (Task 1), `ama/<variante>/predictions.csv` (Task 2), `clips.jsonl` e `features/<variante>/<video>.npz` (`a_prob`).
- Produces, em `report.py`:
  - `best_tiou(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray`: por linha de `a`, o maior tIoU com as linhas de `b` do mesmo `video-id` e `label`; 0 se não há.
  - `agreement(reference, found, tiou=0.5, min_score=0.3) -> dict`: filtra os dois por `score >= min_score` e devolve `{"recall", "precision", "mean_tiou"}`.
    - `recall` é a parcela da referência com `best_tiou >= tiou`.
    - `precision` é a mesma conta para `found`.
    - `mean_tiou` é o `best_tiou` médio sobre a referência (0 para as não encontradas).
    - `None` quando o lado correspondente está vazio: `recall` e `mean_tiou` sem referência, `precision` sem `found`.
  - `class_agreement(dense: np.ndarray, other: np.ndarray) -> float`: parcela de clipes com o mesmo `argmax`; `ValueError` se o número de clipes difere.
  - `window_class(probabilities, starts_s, span_s, window) -> int | None`: classe do `argmax` da média das probabilidades dos clipes inteiramente dentro da janela; `None` se não há.
  - `make_report(out, min_score=0.3, tiou=0.5) -> Path`.
- Colunas do **Resumo**: `Variante`, `Concordância de classe com o denso`, `Detecções do denso reencontradas (recall)`, `Detecções confirmadas pelo denso (precisão)`, `tIoU médio com o denso` (as três últimas só se o AMA rodou), e as de hoje: `Tokens médios por clipe`, `Tokens mantidos (%)`, `Tokens na região do motorista`, `Cosseno médio com o denso`, `Speedup previsto`, `Speedup medido`, `Tempo médio por clipe (s)`. As abas: `Resumo`, `Por detecção do denso`, `Por vídeo`, `Configuração`. Sai a aba `AMA (mAP)`.
- A aba `Por detecção do denso` tem uma linha por detecção do denso acima do limiar: `Vídeo`, `Classe` ("classe N"), `Início (s)`, `Fim (s)`, `Confiança do denso` e, por variante, `<curto>: melhor tIoU` e `<curto>: classe no trecho`.

- [ ] **Step 1: Escrever os testes** em `tests/benchmark/test_report.py` (`pytest.importorskip("pandas")` e `matplotlib`; o teste da planilha também `openpyxl`). O helper `_d(rows)` cria um `DataFrame` com as colunas `video-id, t-start, t-end, label, score`, também quando `rows` é `[]`.
```python
def test_best_tiou_pairs_each_detection_with_its_best_same_class_overlap():
    reference = _d([["v", 10, 20, 3, .9], ["v", 30, 40, 3, .9], ["w", 0, 5, 3, .9]])
    found = _d([["v", 15, 25, 3, .4], ["v", 10, 20, 5, .9], ["v", 12, 20, 3, .1]])
    assert report.best_tiou(reference, found) == pytest.approx([0.8, 0.0, 0.0])

def test_agreement_counts_what_the_variant_finds_and_what_it_invents():
    reference = _d([["v", 10, 20, 3, .9], ["v", 30, 40, 3, .9], ["v", 10, 20, 3, .1]])  # last: below the floor
    found = _d([["v", 10, 21, 3, .8], ["v", 50, 60, 2, .7]])
    got = report.agreement(reference, found, tiou=0.5, min_score=0.3)
    assert (got["recall"], got["precision"]) == (0.5, 0.5)
    assert got["mean_tiou"] == pytest.approx((10 / 11) / 2)

def test_a_variant_that_finds_nothing_has_zero_recall_and_no_precision():
    got = report.agreement(_d([["v", 10, 20, 3, .9]]), _d([]))
    assert (got["recall"], got["precision"], got["mean_tiou"]) == (0.0, None, 0.0)

def test_a_dense_run_without_detections_leaves_recall_undefined():
    got = report.agreement(_d([]), _d([["v", 10, 20, 3, .9]]))
    assert (got["recall"], got["mean_tiou"], got["precision"]) == (None, None, 0.0)

def test_class_agreement_is_the_share_of_clips_with_the_dense_class():
    dense = np.eye(3)[[0, 1, 2, 0]]
    assert report.class_agreement(dense, np.eye(3)[[0, 1, 1, 0]]) == 0.75
    with pytest.raises(ValueError, match="clips"):
        report.class_agreement(dense, dense[:3])

def test_the_class_in_a_window_averages_the_clips_wholly_inside_it():
    probabilities = np.array([[.8, .2], [.4, .6], [.4, .6], [1., 0.]])
    assert report.window_class(probabilities, np.array([0., 1., 2., 3.]), 2.0, (0.0, 4.0)) == 1
    assert report.window_class(probabilities, np.array([0., 1., 2., 3.]), 2.0, (10.0, 12.0)) is None
```
O teste de ponta a ponta monta uma execução sintética em `tmp_path` (`run.json` com `schedules`, `settings.fixed_mask`, `settings.prune_layer`, `weights`, `policy`, `region`, `environment`, `check`; `clips.jsonl` com `start_s`, `tokens`, `inside_region`, `cosine_to_dense`, `seconds`; `features/<variante>/v.npz` com `a_prob`; `selection/v_<start>.npz` com os `*_kept` e os dois `*_prior`; `frames/v/<start>.jpg`; `policy.json` e `region.json`), com e sem `ama/<variante>/predictions.csv`:
```python
def test_the_report_without_labels_compares_every_variant_with_the_dense(tmp_path):
    out = _synthetic_run(tmp_path, with_ama=True)
    book = pd.ExcelFile(report.make_report(out))
    assert book.sheet_names == ["Resumo", "Por detecção do denso", "Por vídeo", "Configuração"]
    summary = book.parse("Resumo").set_index("Variante")
    assert summary.loc["Sem ViT-Prune (denso)", "Detecções do denso reencontradas (recall)"] == 1.0
    assert "Concordância de classe com o denso" in summary.columns
    everything = [str(c) for s in book.sheet_names for c in book.parse(s).columns]
    assert not any("acerto" in c.lower() or "mAP" in c for c in everything)
    assert (out / "figuras" / "resumo.png").exists() and (out / "figuras" / "linha-do-tempo-v.png").exists()

def test_the_report_without_the_ama_still_writes_the_head_and_time_columns(tmp_path):
    out = _synthetic_run(tmp_path, with_ama=False)
    summary = pd.ExcelFile(report.make_report(out)).parse("Resumo")
    assert "Concordância de classe com o denso" in summary.columns
    assert "Detecções do denso reencontradas (recall)" not in summary.columns

def test_a_dense_run_without_detections_still_gives_a_report(tmp_path):
    out = _synthetic_run(tmp_path, with_ama=True, dense_detections=False)
    assert report.make_report(out).exists()

def test_the_report_needs_a_finished_run(tmp_path, capsys):
    assert main(["benchmark", "report", "--out", str(tmp_path)]) == 2
    assert "vitprune benchmark run" in capsys.readouterr().err

def test_the_report_step_takes_its_thresholds_from_its_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(report, "make_report", lambda out, min_score, tiou: seen.update(
        out=out, min_score=min_score, tiou=tiou))
    assert main(["benchmark", "report", "--out", "o", "--min-score", "0.4", "--tiou", "0.6"]) == 0
    assert seen == {"out": "o", "min_score": 0.4, "tiou": 0.6}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `PYTHONPATH=src python3 -m pytest tests/benchmark/test_report.py -o addopts="" -q`
Expected: `AttributeError` das funções novas.

- [ ] **Step 3: Implementar** as funções puras e reescrever `summary`, `actions` (vira a tabela por detecção do denso), `configuration`, `timeline_figure`, `selection_figure` e `summary_figure`.
  - Sai tudo o que leu anotações, `ama.json` e `match`; as classes viram "classe N".
  - A linha do tempo tem uma pista por variante, com a do denso chamada "Denso (referência)", e as cores vêm dos rótulos presentes nas detecções.
  - O título da figura de seleção traz o vídeo e o instante do clipe; o arquivo passa a ser `selecao-<final do vídeo>-<início>.png`.
  - O gráfico-resumo usa recall e concordância de classe no lugar de mAP e "ações reconhecidas".
  - A aba `Configuração` troca as notas de leitura por ação por uma nota da referência: "Referência: o denso. Concordância não é acerto: se o denso errar, a variante que erra igual concorda."
  - `PERCENT` passa a incluir as colunas de concordância, recall e precisão.
  - O handler do CLI usa `--min-score` (0,3) e `--tiou` (0,5), sem `--annotations`.

- [ ] **Step 4: Rodar e ver passar**

Run: `PYTHONPATH=src /tmp/opencode/venv/bin/python -m pytest tests/benchmark -o addopts="" -q --basetemp=/tmp/opencode/t3` e `ruff check src tests`
Expected: tudo passa (o venv de `/tmp/opencode` tem `openpyxl`).

- [ ] **Step 5: Conferir as figuras.** Abrir `linha-do-tempo-v.png` e uma `selecao-*.png` em `/tmp/opencode/t3/` e confirmar que os títulos dizem "Denso (referência)" e "classe N", sem "ação" nem "acerto".

---

### Task 4: documentação e verificação final

**Files:**
- Modify: `README.md`, `docs/operations.md`, `docs/architecture.md`, `benchmark/EXPERIMENTO.md`, `src/vitprune/pipeline/benchmark/__init__.py`, o docstring de `benchmark` em `src/vitprune/pipeline/cli.py`, `benchmark/requirements.txt` (comentário)

- [ ] **Step 1: Atualizar os textos.**
  - `docs/operations.md`, seção `benchmark`:
    - `run` não lê anotações e ganha `--show`.
    - `ama` roda o modelo direto, no interpretador de `--ama-python` (padrão: o atual), e recusa variante já avaliada.
    - `report` compara com o denso, com `--min-score` e `--tiou`.
    - A lista de arquivos perde `ama.json` e `labels.json` e ganha `ama/<variante>/predictions.csv` e `ama-manifest.json`.
    - A frase "a concordância não é acerto" entra.
  - `README.md`, `docs/architecture.md`, o `__init__.py` do pacote e o docstring do CLI: tirar "annotated videos" e `--annotations`.
  - `benchmark/EXPERIMENTO.md`: as perguntas passam a ser sobre concordância e tempo; a linha "Anotação" de Dados vira "Sem anotação: a referência é o denso"; os comandos perdem `--annotations` e usam `--ama-python`.

- [ ] **Step 2: Procurar sobras**

Run: `grep -rn "class-intervals\|--annotations\|ama\.json\|mAP\|acerto\|entry-point" src docs/operations.md docs/architecture.md README.md benchmark/EXPERIMENTO.md`
Expected: só aparecem os usos intencionais (as notas "não é acerto" e o `--entry-point` do `vitprune evaluate`).

- [ ] **Step 3: Suíte completa**

Run: `PYTHONPATH=src /tmp/opencode/venv/bin/python -m pytest tests -o addopts="" -q` e `ruff check src tests`
Expected: tudo passa, exceto `test_public_library_imports_do_not_load_pipeline_extras`, que já falhava antes: o teste roda com `-I` e não vê o `torch` de `~/.local`.

- [ ] **Step 4: Relatar** ao dono o que foi conferido com o AMA real (Task 2, Step 5) e o que continua sem verificação: o `benchmark run` com o VideoMAE real e o `benchmark ama` no ambiente do AMA da outra máquina.
