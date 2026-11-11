# Relatório — FP16 + SDPA e fusão progressiva

Branch `exp/fp16-sdpa-progressive` (base `12eb727`). Spec e plano locais:
`docs/superpowers/specs/2026-11-11-fp16-sdpa-progressive-design.md` e
`docs/superpowers/plans/2026-11-11-fp16-sdpa-progressive.md`. Os resultados ficam em
`benchmark/results/fp16-sdpa/`, fora do Git. Cada pasta tem o `commit.txt` do código executado.

## Condições comuns

| Item | Valor |
|---|---|
| Máquina | Jetson AGX Orin (MAXN), CUDA 12.6, torch 2.11 |
| Modelo | VideoMAEv2-g `videomae-v2_inference.pth` (`58cbaf65…`). Arquivos do VideoMAEv2 intactos |
| Vídeos (dev) | `13522_NoAudio_7`, `31903_NoAudio_7` e `33682_NoAudio_5`: 2.763 clipes, 48 ações. Pasta `motion/videos/u896-dev` |
| Amostragem | stride 16, `frame_step` 2, lote 8, corte após o bloco 2, `motion_weight` 0,5 |
| Box | `regiao-ajuste.json` (`6ad49d77…`), a mesma em todos os braços com poda. Calibrada nos motoristas de dev (`--allow-region-overlap`, registrado) |
| AMA | inalterado, labels `label_A1-train_A1-val.json`, corte 0,30 e tIoU 0,5 |
| Holdout / reserva | não usados |

**Ressalva que continua valendo:** os vídeos de dev estão no treino do AMA, e a box foi calibrada nesses
motoristas. Os números medem o quanto cada braço preserva o pipeline, não a generalização.

## 1. Opção 2 — FP16 (`autocast`) + SDPA

Opções explícitas novas:
- `PruningPlan(attention="sdpa")` e `benchmark run --attention sdpa`. Rodam cada bloco por
  `vitprune.core.attention.run_block`, que lê os pesos do próprio bloco e chama
  `F.scaled_dot_product_attention`.
- A precisão continua em `--precision fp16`.

### 1.1 Checagem curta (C2): 48 clipes, o 1º batch e o do meio de cada vídeo

Comparação contra o denso FP32 com a atenção original (`c2/check.json`):

| Configuração | Cosseno P10 / P50 / mín | max \|Δ\| prob. | Top-1 = denso FP32 | Índices mantidos = FP32 |
|---|---|---:|---:|---:|
| Denso FP32 + model | 1 / 1 / 1 | 0 | 100% | — |
| Denso FP16 + model | 0,99999 / 0,99999 / 0,99982 | 0,008 | 100% | — |
| Denso FP16 + SDPA | 0,99999 / 0,99999 / 0,99994 | 0,005 | 100% | — |
| U-1024 FP32 + model | 0,50 / 0,70 / 0,44 | 0,70 | 62,5% | 100% |
| U-1024 FP16 + model | 0,47 / 0,71 / 0,44 | 0,69 | 60,4% | 75% |
| U-1024 FP16 + SDPA | 0,47 / 0,72 / 0,45 | 0,69 | 62,5% | 75% |

- **Gate:** passou. Não houve NaN/Inf, e o denso FP16+SDPA ficou com cosseno P10 ≥ 0,999.
- **Seleção no U-1024:** em FP16, 25% dos clipes mudam algum índice mantido, por quase-empates do
  score. O efeito no embedding é da mesma ordem do da própria poda.
- **Kernel executado** (`c2/sdpa_backend.json`):
  - O `torch.profiler` não registra eventos CUDA neste Jetson: o CUPTI exige root.
  - A confirmação veio do dispatcher (`torch._fused_sdp_choice`) e de cada backend forçado, nas
    formas de um bloco (8 × 16 cabeças × N × 88):
    - **FP16 → FlashAttention.** A saída é bit a bit igual à da chamada padrão: 14,5 ms em N=2048,
      contra 28,3 ms do memory-efficient e 219 ms do math.
    - **FP32 → memory-efficient.** Flash não existe em FP32.

### 1.2 Execução completa B1 (FP16+SDPA) contra o FP32 salvo (`motion/u896-r1`)

| Braço | Recuperadas | FP | mAP médio | p50 s/clipe | p95 |
|---|---:|---:|---:|---:|---:|
| Denso FP32 (u896-r1) | 48/48 | 2 | 1,000 | 2,062 | 3,223\* |
| U-1024 FP32 (u896-r1) | 46/48 | 4 | 0,977 | 0,950 | 1,488\* |
| **Denso FP16+SDPA** | **48/48** | **2** | **1,000** | **0,326** | 0,462 |
| **U-1024 FP16+SDPA** | **46/48** | **4** | **0,977** | **0,153** | 0,218 |

\* O p95 do u896-r1 inclui um trecho com a sessão ativa (contaminação já registrada).

- **Ação por ação, denso:** o FP16+SDPA reencontra exatamente as mesmas 48 ações que o FP32, com
  max |Δscore| = 0,0003.
- **Ação por ação, U-1024:** perde as mesmas 2 ações que o FP32 (13522_7 classe 5 e 31903_7 classe 6).
  O max |Δscore| é 0,034, por causa das seleções que mudam em FP16 (§1.1).
- **Repetição em B2:** o denso FP16+SDPA tem scores idênticos, e o U-1024 fica a até 0,0004, por causa
  dos atômicos do merge.
- **Critério "preserva a qualidade"** (congelado antes): **satisfeito.** Isso liberou a B2.

## 2. Opção 3 — fusão progressiva

`PruningPlan(stages=((10, 1024), (20, 768)))` e `benchmark run --progressive 1536 10:1024 20:768`
(variante `prog_1536_1024_768_mascara`):

1. **Etapa 1, igual ao U-1024:** depois do bloco 2, o score com box mantém 1536 tokens (192 por slice),
   e o resto funde no mais parecido.
2. **Depois do bloco 10, → 1024 tokens; depois do bloco 20, → 768.** O clipe inteiro passa por
   *bipartite soft matching*, como no ToMe (arXiv 2210.09461), entre todos os slices:
   - A/B alternados;
   - as arestas de maior cosseno fundem por média ponderada pelo `size`;
   - os `size` se somam, e o pooling ponderado continua igual à média densa no instante de cada fusão.
3. **Chaves:**
   - `k = LN1(x) · W_kᵀ` do **bloco seguinte**, com `qkv.weight[D:2D]`;
   - o VideoMAE não tem viés de chave;
   - média sobre as 16 cabeças.

   É a projeção que esse bloco calcularia, calculada uma vez a mais. Cada etapa custa ≈ `2·N·D²` FLOPs,
   mais o casamento de `N/2 × N/2 × 88`. Isso roda dentro de `encode()` e entra em toda latência
   abaixo. O casamento é feito em float32, inclusive sob `autocast`.
4. **Atenção proporcional desligada**, para isolar o efeito.

**Congelamento do cronograma** (`latency-fp32-try1/latency.json`):
- FP32 com a atenção original, 48 clipes, 2 aquecimentos, 5 passes, ordem alternada;
- U-1024: p50 1.088,9 ms; P: p50 1.120,0 ms;
- razão 1,029, dentro de ±5% → aceito na 1ª tentativa.

O cronograma foi registrado no adendo da spec antes de qualquer AMA com P. Pela conta analítica, a média
é de 1.049,6 tokens por bloco, contra 1.075,2 do U-1024.

### 2.1 Execução A — FP32, atenção original

_(pendente)_

### 2.2 Execução B2 — FP16+SDPA (mesmo cronograma)

| Braço | Recuperadas | FP | mAP médio | p50 s/clipe | p95 |
|---|---:|---:|---:|---:|---:|
| Denso | 48/48 | 2 | 1,000 | 0,358 | 0,367 |
| U-1024 | 46/48 | 4 | 0,977 | 0,172 | 0,177 |
| **P 1536→1024→768** | **46/48** | **2** | **0,990** | **0,166** | **0,172** |

- **Perdas:** P perde as mesmas 2 ações que o U-1024.
- **FPs:** P fica com 2 FPs, como o denso:
  - classe 8 em 124–143 s, que o denso também tem (a confusão conhecida 5→8);
  - classe 10, que o U-1024 também tem.

  Some o FP de classe 12 e o de classe 9 do U-1024.
- **Nas 46 ações em comum** (denso / U / P):

  | | Denso | U | P |
  |---|---:|---:|---:|
  | Score médio | 0,844 | 0,791 | 0,800 |
  | Erro de início | 0,60 s | 0,71 s | 0,59 s |
  | Erro de fim | 0,66 s | 0,66 s | 0,68 s |

  - Na diferença pareada de score P − U, a mediana é 0,000, e P fica maior em 22 de 46: não há
    deslocamento sistemático.
- **Ação frágil 13522_7, classe 6:** U 0,31 (a 0,01 do corte), contra P 0,49.

## 3. Latência medida (`arms_latency.py`: 48 clipes, lote 8, 2 aquecimentos, 5 passes alternados; ms/clipe)

| Modo | Denso | U-1024 | P | P / U |
|---|---:|---:|---:|---:|
| FP32 + atenção original | _(pendente)_ | 1.088,9 | 1.120,0 | 1,029 |
| FP16 + SDPA (Flash) | 355,2 | 169,1 | 163,3 | 0,966 |

O custo relativo das fusões muda entre os modos. Em FP32, P fica 3% mais lento que o U-1024. Em
FP16+SDPA, fica 3% mais rápido. Isso não foi decomposto. Uma hipótese é que, com a atenção fundida, o
custo fixo por token dos blocos 3–10 pese menos frente à economia dos 768 tokens nos blocos 21–40.

## 4. Leitura

_(pendente da execução A)_
