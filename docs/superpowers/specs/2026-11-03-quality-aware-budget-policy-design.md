# Quality-Aware Budget Policy (QABP) com probe de compressibilidade

Base: `ViTPrune_Handoff_QualityAware_Compressibilidade.md` (handoff de 2026-11-03), branch
`experiment/quality-aware-policy` a partir de `benchmark` (`20f28e1`).

## Intenção

A política legada (`BudgetPolicy`: ranks de movimento e redundância contra 101 quantis,
`g = ½·rank(M) + ½·(1 − rank(R))`, tercis de `g` → 256/512/768) deu 256 tokens a 60–70% dos
clipes dos dois vídeos avaliados. O embedding final ficou longe do denso: cosseno médio 0,17;
por nível, ~0,0 em 256, 0,28–0,48 em 512 e ~0,77 em 768 (`benchmark/results/clips.jsonl`).

A QABP troca a pergunta "quanto movimento este clipe tem em relação à câmera?" por "qual o
menor orçamento cuja perda de fidelidade prevista é aceitável?". Ela aprende offline, em vídeos
de calibração, a ligação entre sinais medidos no corte (após o bloco 2) e o drift do embedding
final em relação ao VideoMAEv2 denso. **Fidelidade tem precedência sobre speedup.**

## Objetivos

1. Diagnóstico P0 antes da política: 2048 tokens reproduz o denso; curva 512/1024/1536/2048
   com scoring e merge atuais; drift por etapa; erro por token no merge; latência por nível.
2. QABP funcional: probe barato após o bloco 2, tabela de risco calibrada com quantil alto,
   escolha do menor nível admissível, fallback denso observável, schema próprio, versionado e
   que falha fechado.
3. Ablação B (sem probe) com a mesma infraestrutura.
4. Validação no holdout de calibração, separado por motorista: correlação probe × drift
   final, cobertura da tolerância, custo.
5. Benchmark novo nos dois vídeos avaliados: denso, legacy, fixo 1024, fixo 1536, QABP e QABP
   sem probe, todos sem máscara, com AMA e relatório.
6. Medir o custo do probe e o speedup real, não só a redução de FLOPs.

## Não-objetivos

- Mudar scoring, priors, merge, posição do corte ou atenção, salvo bug comprovado no P0.
- Retreinar ou ajustar VideoMAEv2, cabeça de 16 classes ou AMA.
- Repetir o estudo de máscaras ou usar prior espacial na QABP.
- Regressores, bins 2D, mais níveis, varredura de hiperparâmetros, JS das probabilidades.

## Anti-objetivos (seria falha mesmo funcionando)

- Calibrar ou ajustar qualquer coisa olhando os dois vídeos avaliados.
- Dar a QABP por validada porque os testes passam: se o probe não correlacionar de forma
  significativa com o drift final no holdout, o relatório diz isso.
- Fallback que esconde erro estrutural: schema, checkpoint ou contexto incompatível é erro,
  não "orçamento denso".
- Rodar o denso, ou continuar os blocos 3–40 mais de uma vez, na inferência online.
- Anunciar melhoria sem medição nas mesmas condições.

## Restrições

- Camadas (`tests/test_layering.py`): `core` só importa `core`; `bench`, `core`; `pipeline`,
  `core` e `bench`.
- Sem dependências novas: Spearman e teste de permutação com numpy.
- `BudgetPolicy`, `calibrate_policy`, `vitprune calibrate` e `policy.json` intocados e
  reproduzíveis; API e CLI existentes continuam funcionando, e os 495 testes atuais passam.
- Ambiente das medições: Jetson AGX Orin, CUDA, FP32 (`--precision off`), batch 8,
  `prune_layer` 2, stride 16, `frame_step` 2, pesos `videomae-v2_inference.pth`
  (sha256 `58cbaf65…`).
- Níveis são múltiplos das 8 fatias; o nível 2048 é o caminho denso de verdade (sem merge).

## Abordagens

- **A — MVP do handoff com probe (escolhida).** Probe de âncoras, índice `h`, tabela de risco
  por faixas de `h`, fallback denso. É auditável, barata online e entrega o que o handoff pede.
- **B — Mesma tabela sem probe (`h = rank(g)`).** Não é o produto: entra como ablação, com o
  mesmo código (parâmetro `alpha = 1`), na calibração e no benchmark.
- **C — Regressão contínua do drift por nível (descartada).** Usa melhor amostras pequenas,
  mas é menos auditável, e o handoff a desaconselha no primeiro ciclo.

## Dados e split (fixados antes de qualquer medição)

| Conjunto | Vídeos |
|---|---|
| Avaliados (benchmark) | `38159_NoAudio_7`, `38479_NoAudio_5` |
| Fora da QABP | os dois avaliados e `38159_NoAudio_5` (mesmo motorista de um avaliado) |
| Ajuste | `13522_7`, `28557_5`, `28557_7`, `30932_5`, `30932_7`, `31903_5`, `31903_7`, `33682_5`, `33682_7` (5 motoristas) |
| Holdout | `35418_5`, `35418_7`, `36305_3`, `36305_5` |

O holdout tem os dois motoristas de id maior, regra escolhida antes de medir. São 48 clipes por
vídeo, espalhados com `spread_starts`: 432 clipes de ajuste e 192 de holdout. (Os nomes acima
omitem o prefixo `Dashboard_user_id_` e o infixo `_NoAudio_`.)

A legacy continua sendo o `policy.json` atual, como no benchmark anterior. Ele foi calibrado
com os 16 vídeos, **incluindo os dois avaliados**. Isso vira ressalva no relatório e não é
corrigido, para o baseline continuar reproduzível.

## Desenho

### P0 — diagnóstico (primeira coisa a rodar)

- **Painel:** 12 clipes por vídeo avaliado, espalhados (`spread_starts`), 24 no total. Para
  cada clipe registro também o regime legado que ele teve.
- **Medidas por clipe e nível:**
  - cosseno e L2 relativo do embedding final contra o denso;
  - concordância top-1 da cabeça;
  - erro por token no merge: `1 − cos(token original, representante)`, com média, p90 e máximo;
  - cosseno dos vetores resumidos depois dos blocos 2 (logo após o merge), 3, 6, 10, 20 e 30;
  - os sinais de entrada da política.
- **Latência:** `measure_plans` com warmup e sincronização CUDA, com planos denso e fixos
  512/1024/1536/2048, em um batch de 8 clipes reais. Roda por um script curto de
  diagnóstico em `benchmark/`; não ganha comando novo.
- **Identidade:** `check_backbone` e embedding fixo 2048 × denso no mesmo batch.

Os registros do P0 ficam em `benchmark/results/qabp/p0/` e nunca entram na calibração.

Gates pré-registrados:

| Condição | Ação |
|---|---|
| G1: `max |E(2048) − E(denso)| > 1e-4`, ou `check_backbone` falha | parar, corrigir o mínimo, registrar a causa |
| G2: P50 do cosseno em 1536 < 0,90 | parar e voltar a você antes da política (a auditoria de merge e corte é outro projeto) |
| Caso contrário | seguir com a QABP e `K = {512, 1024, 1536, 2048}` |

Depois da calibração, antes do benchmark completo:

| Condição | Ação |
|---|---|
| G3: com ε = 0,10, qualquer um entre V0 (degenerada), V1 (probe) e V2 (cobertura) falha no holdout | parar; não rodar o benchmark completo; não mudar ε sozinho; entregar o relatório diagnóstico do G3 e esperar a sua decisão |
| Caso contrário | congelar a política e rodar o benchmark completo |

Relatório diagnóstico do G3, só com dados de calibração (ajuste e holdout), nunca com os
vídeos avaliados:
- distribuição dos orçamentos escolhidos e dos motivos (`ok`, `no_level_within_tolerance`,
  fallbacks);
- cobertura, drift (`d_cos`: média, P50, P90) e custo analítico estimado (`budget_cost`);
- comportamento do probe: ρ e p por nível, ρ por vídeo, e o mesmo para `g` e `h`;
- curva exploratória para ε ∈ {0,10; 0,15; 0,20; 0,30}: tokens médios, % abaixo de 2048,
  cobertura, `d_cos` e speedup previsto, para A, B e os níveis fixos.

Se V1 falhar, investigo a causa antes de apresentar. Olho, por exemplo, se o erro do probe
acompanha o erro por token do merge, se o drift é dominado pela amplificação nos blocos
seguintes (cos por etapa) e se o sinal existe dentro de cada vídeo mas some entre vídeos.
Aumentar ε para forçar compressão não é resposta.

### Probe — `core/compressibility.py` (novo)

- **Entradas:** a grade no corte `Z [B, 8, 256, 1408]` (antes de qualquer merge) e os scores
  base sem prior.
- **Âncoras (`q = 16` por fatia):**
  - 8 posições fixas espalhadas: `round(linspace(0, P−1, 8))` no índice raster;
  - os 8 maiores scores base, excluindo essas posições.
  A seleção é determinística.
- **Erro por token:** `e[t,p] = 1 − max_a cos(z[t,p], z[t,a])`, em float32, contra as âncoras
  da própria fatia, sem materializar `P × P`.
- **Features do clipe:** `probe_mean = mean(e)` e `probe_p90 = p90(e)` sobre os 2048 tokens,
  sem min-max por fatia. Tokens de norma zero dão `e = 1`; tokens não finitos dão feature NaN.
- **Retorno:** `CompressibilityFeatures(motion, redundancy, probe_mean, probe_p90)`, cada um
  `[B]`. O movimento e a redundância vêm de `clip_statistics`, sem mudança.
- **Custo:** `S·P·q·D ≈ 46 MMAC` por clipe, mais um top-k e um quantil. O tempo é medido
  (ver Benchmark).

### Política — `core/policy.py`

**`QualityAwareBudgetPolicy`** (frozen, serializável, `policy_type = "quality_aware_v1"`, versão
por hash do conteúdo, como a legacy).

Campos:
- `levels`, com o último igual a `geometry.num_tokens`;
- referências de 101 quantis de M, R, `g` e `c = probe_p90`;
- `motion_balance = 0.5`, para `g`;
- `alpha` (0,5 em A, 1,0 em B);
- `anchors = 16`;
- `bin_edges` (5 cortes → 6 faixas de mesma contagem no ajuste);
- `risk[faixa][nível < 2048]`;
- `support` e `support_videos` por faixa;
- `risk_quantile = 0.9`, `tolerance = ε`, `min_support = 30`, `min_videos = 3`;
- `feature_range`: o mínimo e o máximo de M, R e c no ajuste.

**`choose_budget(features) -> BudgetDecision`**, por clipe:

1. Feature usada não finita → 2048, `reason = non_finite`, fallback.
2. Feature usada fora de `feature_range` → 2048, `out_of_range`, fallback.
3. Calcula `h = alpha·rank(g) + (1 − alpha)·rank(c)` e acha a faixa. Faixa com suporte menor
   que 30 clipes ou 3 vídeos → 2048, `low_support`, fallback.
4. Escolhe o menor `k < 2048` com `risk[faixa][k] ≤ ε`; se não houver → 2048,
   `no_level_within_tolerance` (decisão normal, não conta como fallback).

`BudgetDecision` traz:
- `tokens`, `regime` (índice no nível) e `predicted_risk` (0 para 2048), todos `[B]`;
- `support` e `used_fallback`, também `[B]`;
- `reason` e as `features`.

Quando `alpha = 1`, o probe não roda.

**`calibrate_quality_aware(records, levels, tolerance, alpha, …)`:**
- referências e faixas vêm só dos clipes de ajuste;
- o risco é o quantil 0,9 (método `higher`) de `d_cos = 1 − cos(E_denso, E_k)` em cada faixa;
- falha se as faixas não forem estritamente crescentes.

**`FixedBudget(tokens)`:** o mesmo número de tokens para todos os clipes. Serve aos baselines
fixos e ao P0.

**`ε = 0,10`:** no P90 de cada faixa, cosseno ≥ 0,90. Fica fixado antes de medir.

### Encoder — `core/encoder.py`

- `PruningPlan.budget` aceita `BudgetPolicy`, `QualityAwareBudgetPolicy` ou `FixedBudget`.
- A QABP guarda a seleção com que foi calibrada: `motion_weight`, sem prior espacial e sem
  atenção proporcional. `PruningPlan` recusa (falha fechado) uma QABP com outro
  `motion_weight`, com `spatial_prior` ou com `size_biased_attention`, porque a tabela de
  risco não vale para outra seleção. Combinar com máscara exige recalibrar; fica para um
  ciclo futuro.
- Após o corte:
  - `base = score_tokens(grid, motion_weight)`, sem prior;
  - `scores = apply_prior(base, prior)`, com `apply_prior` tornado público em `scoring.py`
    (sem mudar o resultado);
  - a decisão usa estatísticas, probe e `base`; a seleção usa `scores`; o merge não muda.
- Grupo com `keep == tokens_per_slice` segue sem merge, pelo caminho denso: tokens originais,
  `sizes = None` e mean pooling.
- `ClipBudget` ganha `decision: BudgetDecision | None = None`, como campo final com default.
  Isso quebraria quem desempacota três campos, mas nada no repositório faz isso.
- **`encode_levels(clips, levels, depths=())`**, offline:
  - roda até o corte uma vez e continua cada nível a partir da mesma grade;
  - devolve as features, os embeddings por nível, os vetores resumidos nas profundidades e o
    erro por token do merge;
  - `merging.merge_error()` é público e testado.
  É o único caminho de medição do P0 e da calibração.

### Calibração offline

**`bench/quality.py`:** a partir de `encode_levels`, monta um registro por clipe.

Cada registro tem:
- `video`, `start`, M, R, `probe_mean`, `probe_p90`;
- por nível: `d_cos`, `d_rel`, `top1`, erro do merge e cos por etapa.

O módulo também traz:
- `spearman(x, y)`;
- `permutation_p(x, y, groups, n=10000, seed=0)`: unilateral, permutando dentro de cada vídeo;
- `time_probe(...)`: tempo do probe e de `clip_statistics` por clipe, com warmup e
  sincronização, num tensor do tamanho real (o custo depende só da forma).

**`pipeline/quality.py`:**
- **Coleta:** vídeos → clipes espalhados → `records/<stem>.jsonl`, escritos inteiros ao fim
  de cada vídeo, mais `records/settings.json`. Retoma por vídeo e recusa pasta com outras
  configurações. Registra o sha256 de cada vídeo.
- **Ajuste:** lê os registros, separa o holdout pela lista de stems, ajusta A e B e grava o
  arquivo da política pedida (A, ou B com `--no-probe`). Para a QABP, `read_policy` confere
  precisão, corte, `frame_step` e checkpoint. A seleção é conferida pelo `PruningPlan` (ver
  Encoder). O arquivo é
  `{"policy": …, "calibration": …, "validation": …}`. `calibration` traz:
  - schema e checkpoint;
  - corte, stride, `frame_step` e precisão;
  - `motion_weight`, âncoras e geometria;
  - vídeos de ajuste e de holdout, com sha256.
- **`read_policy`** (em `pipeline/calibration.py`):
  - despacha por `policy_type`: ausente → legacy; `quality_aware_v1` → QABP; outro → erro;
  - para a QABP, confere também a precisão quando informada;
  - todo comando que já aceita `--policy` passa a aceitar os dois.

### CLI

- `vitprune budget-curve`:
  - parâmetros: `--videos` (um ou mais arquivos ou pastas), `--clips-per-video`, `--levels`,
    `--depths`, `--out <pasta de registros>`, mais os argumentos de modelo e execução atuais;
  - imprime P10/P50/P90 do cosseno por nível;
  - serve ao P0 e à coleta.
- `vitprune calibrate-quality`:
  - parâmetros: `--records`, `--holdout STEM…`, `--tolerance 0.1`, `--bins 6`, `--no-probe` e
    `--out` (recusa sobrescrever);
  - imprime a validação no holdout;
  - não usa GPU.
- `vitprune calibrate` continua como está, para a legacy.

### Benchmark

- `vitprune benchmark run` ganha `--quality-policy`, `--ablation-policy` e `--fixed K…`. Com
  `--quality-policy`, o estudo vira o de política: `denso`, `vitprune` (legacy sem máscara),
  `fixo_1024`, `fixo_1536`, `qabp` e `qabp_sem_probe`. Sem ele, o estudo de máscaras continua
  idêntico. `--region` continua obrigatório; no estudo de política, a região só mede a
  fração de tokens mantidos dentro dela.
- O `run.json` registra:
  - as variantes e os arquivos de política;
  - o tempo do probe por clipe (`time_probe`);
  - a sobreposição de vídeos com a legacy.
- **Guarda de split:** recusa uma QABP cujos vídeos de calibração (por sha256) incluam um
  vídeo avaliado. Para a legacy, só avisa e registra.
- `clips.jsonl` ganha `rel_l2_to_dense`, `predicted_risk` e `reason` (nulos quando não se
  aplicam).
- O relatório lê as variantes do `run.json`. O Resumo por variante traz:
  - tokens médios e % por nível;
  - cosseno médio, P10 e P50, e L2 relativo;
  - concordância de classe e recall, precisão e tIoU do AMA contra o denso;
  - speedup previsto (`budget_cost`), tempo p50/p95 por clipe e speedup medido;
  - % de fallback.
- A aba nova "QABP" traz, por variante QABP: o Spearman entre risco previsto e `1 − cos`, a
  cobertura `1 − cos ≤ ε`, os motivos de decisão e o tempo do probe.
- As figuras de máscara e de seleção só saem no estudo de máscaras.
- Saídas em `benchmark/results/qabp/bench/`. Nada em `benchmark/results/` é sobrescrito.

### Validação (pré-registrada; nada é ajustado olhando os vídeos avaliados)

- **V0 — política não degenerada:** pelo menos 20% dos clipes de holdout recebem um nível
  < 2048. Abaixo disso, a política é "degenerada": V1 e V2 viram "não aplicável", e a QABP
  não é declarada validada.
- **V1 — sinal do probe:** no holdout, ρ de Spearman entre `probe_p90` e `d_cos@k` para cada
  `k < 2048`, sobre todos os clipes (cada clipe tem `d_cos` medido em todos os níveis), com p
  de permutação dentro de cada vídeo. Também ρ por vídeo e o mesmo para `g` e `h`. "Níveis
  que a política usa" são os níveis < 2048 escolhidos para pelo menos 10% dos clipes de
  holdout. **Cada um** precisa de ρ > 0 e p < 0,05; senão o relatório diz "probe não
  validado", e a QABP não é declarada validada.
- **V2 — risco calibrado:** cobertura = fração dos clipes de holdout **enviados a um nível
  < 2048** cujo `d_cos` observado nesse nível é ≤ ε. Os clipes densos não entram, para 2048
  não passar sozinho. Deve ser ≥ 0,85, e o P90 previsto × observado é comparado por faixa.
- **V3 — valor:** no holdout, A contra B e contra os níveis fixos, em tokens médios e
  cobertura, mais a curva para ε ∈ {0,10; 0,15; 0,20; 0,30}. A curva só informa; a escolha
  de ε, se mudar, é sua (G3).
- **No benchmark:** a tabela da seção 7 do handoff, com a mesma métrica de risco previsto ×
  observado. É só um relato; não ajusta nada.
- **Ressalvas:** clipes de um mesmo vídeo não são independentes, e o holdout tem 4 vídeos de 2
  motoristas; os p-valores são indicativos.

O benchmark completo só roda se a política passar no G3. Se não passar, paro com o relatório
diagnóstico, e a configuração final é decisão sua.

### Entregáveis e reprodutibilidade

- Commits pequenos na branch, com testes de CPU no backbone de referência. Os testes cobrem
  os sete itens P0 da seção 5 do handoff e mais:
  - o ajuste com registros sintéticos;
  - fallback, OOD e NaN;
  - política editada ou incompatível;
  - a guarda de split;
  - o estudo de política no `benchmark run` e no relatório.
- `benchmark/run_qabp.sh` com os comandos exatos.
- Um relatório em `benchmark/RELATORIO-QABP.md` no formato da seção 7 do handoff.
- `docs/` atualizado: método, operações e contratos.
- Tempo de GPU estimado: P0 ~10 min; coleta ~55 min; benchmark ~4,5 h (6 variantes) mais AMA
  e relatório.

## Fica para quem construir

- Nomes internos, textos de ajuda e layout exato das abas e colunas.
- Como dividir a geração do relatório.
- O formato interno do cache, desde que retome por vídeo e recuse configuração diferente.

Escolhas minhas, a conferir:
- `ε = 0,10`;
- 6 faixas, com mínimo de 30 clipes e 3 vídeos por faixa;
- 48 clipes por vídeo;
- holdout = motoristas 35418 e 36305;
- o gate G2 em 0,90;
- dois baselines fixos, 1024 e 1536;
- a QABP sem probe também no benchmark (+~40 min);
- as regras V0 (20%) e V1 (níveis com ≥ 10% dos clipes);
- QABP com máscara ou com atenção proporcional é recusada nesta rodada.
