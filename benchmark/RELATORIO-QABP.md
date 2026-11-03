# Relatório — Quality-Aware Budget Policy (QABP)

Branch `experiment/quality-aware-policy`. Cada seção traz as condições em que foi medida.
Os números vêm de `benchmark/results/qabp/` (fora do Git); os comandos estão em
`benchmark/run_qabp.sh`.

## 1. P0 — diagnóstico em clipes de calibração (gate de desenvolvimento)

**Situação: G1 passou e G2 não passou. A execução parou aqui, antes de calibrar a QABP,
como ficou registrado no protocolo.**

### Condições

| Item | Valor |
|---|---|
| Comando | `bash benchmark/run_qabp.sh p0` |
| Código | commit `415938f`. Os caminhos usados (denso, `FixedBudget`, `encode_levels`) são os desse commit. |
| Máquina | Jetson AGX Orin, CUDA 12.6, torch 2.11, FP32 (`--precision off`), batch 8 |
| Modelo | VideoMAEv2-g, `videomae-v2_inference.pth` (sha256 `58cbaf65…`), corte após o bloco 2 |
| Amostragem | stride 16, `frame_step` 2, 3 clipes espalhados por vídeo |
| Painel | os 9 vídeos de ajuste (motoristas 13522, 28557, 30932, 31903, 33682), **26 clipes** |
| Seleção | scoring motion-aware (peso 0,5), sem prior, merge atual, sem atenção proporcional |

O painel tem 26 clipes, e não 27, porque `13522_NoAudio_7` rendeu 2 clipes em vez de 3: a
contagem de quadros declarada pelo contêiner passa do último quadro decodificável. O
registro guarda o que existe e não completa a contagem.

Nenhum vídeo de holdout ou de avaliação entrou no P0, e nada aqui ajusta hiperparâmetros.

### G1 — identidade: passou

- `check_backbone`: o caminho denso reproduz `forward_features` (diferença máxima `0`).
- Com o nível de 2048 tokens, que segue o caminho denso sem merge, a diferença para o denso
  no mesmo batch de 8 clipes também foi `0`.

O caminho podado está integrado corretamente; o problema abaixo não é bug de integração.

### G2 — fidelidade × orçamento: não passou

| Tokens | Cosseno P10 | P50 | P90 | Média | L2 rel. médio | Top-1 = denso | Cosseno ≥ 0,90 | Erro do merge por token (média) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | −0,071 | 0,111 | 0,663 | 0,210 | 1,242 | 31% (8/26) | 0/26 | 0,108 |
| 1024 | 0,281 | 0,522 | 0,828 | 0,559 | 0,916 | 42% (11/26) | 2/26 | 0,049 |
| 1536 | 0,557 | **0,739** | 0,959 | 0,751 | 0,673 | 69% (18/26) | 6/26 | 0,013 |

O G2 exigia P50 ≥ 0,90 em 1536 tokens e obteve 0,739. Mesmo mantendo 75% dos tokens, o
embedding final de metade dos clipes fica com cosseno abaixo de 0,74 em relação ao denso, e
só 6 dos 26 clipes chegam a 0,90.

Figuras: `p0/p0-drift.png` (distribuição por nível) e `p0/p0-stages.png` (evolução por
bloco). A tabela por clipe está em `p0/p0-clips.csv`.

### Onde o desvio nasce

Mediana do cosseno, e do L2 relativo, entre a média dos tokens podados (ponderada por
`size`) e a média dos tokens densos, depois de cada bloco, antes de qualquer normalização:

| Depois do bloco | 512 | 1024 | 1536 |
|---:|---:|---:|---:|
| 2 (logo após o merge) | 1,000 (0,00) | 1,000 (0,00) | 1,000 (0,00) |
| 3 | 0,983 (0,19) | 0,992 (0,13) | 0,998 (0,07) |
| 6 | 0,980 (0,21) | 0,991 (0,14) | 0,998 (0,07) |
| 10 | 0,981 (0,20) | 0,992 (0,13) | 0,998 (0,07) |
| 20 | 0,744 (0,70) | 0,859 (0,52) | 0,958 (0,29) |
| 30 | 0,512 (1,13) | 0,678 (0,86) | 0,841 (0,57) |
| 40 | 0,189 (1,23) | 0,550 (0,94) | 0,731 (0,69) |

**Leitura:**
- Logo após o merge as médias coincidem por construção. É a armadilha que o handoff aponta:
  isso não diagnostica nada.
- Até o bloco 10 o resumo quase não se mexe. Em 1536 tokens o cosseno fica em 0,998, e o
  erro por token do merge é pequeno (1,3% em média).
- O afastamento nasce entre os blocos 10 e 40.
- A razão entre as normas fica perto de 1, ou seja, não há colapso de norma.

Na tabela P0.3 do handoff, este é o caso "erro local pequeno, drift final grande": o proxy
local é insuficiente, ou há amplificação nos blocos seguintes.

### Sinais (exploratório: n = 26, 5 motoristas; não é validação)

ρ de Spearman com `d_cos = 1 − cos` em cada nível:

| Sinal | 512 | 1024 | 1536 |
|---|---:|---:|---:|
| `probe_p90` | 0,23 | 0,01 | −0,05 |
| `probe_mean` | −0,19 | 0,05 | −0,10 |
| Erro do merge (média por clipe) | −0,47 | −0,14 | −0,31 |
| Movimento (M) | −0,53 | −0,67 | **−0,80** |
| Redundância (R) | 0,06 | 0,18 | 0,29 |

- **Neste painel, o probe não tem sinal.** Além disso, o `probe_p90` varia pouco entre os
  clipes (0,40 a 0,45).
- **O movimento tem relação forte e inversa com o drift: os clipes estáticos se afastam
  mais.** A política legada faz o oposto: dá menos tokens a quem tem menos movimento, ou
  seja, justamente aos clipes mais frágeis. Isso é coerente com o resultado dela nos vídeos
  avaliados, onde 60–70% dos clipes ficaram com 256 tokens e o cosseno médio foi 0,17.
- **Hipótese, ainda não testada:** num clipe estático, muitos tokens quase idênticos se
  fundem em poucos tokens de `size` grande. Sem atenção proporcional, cada um desses tokens
  ainda pesa como um único token na atenção, e a distorção se acumula nos blocos profundos.
  O código já tem a atenção proporcional (`size_biased_attention`, experimental e
  desligada).

### Latência por nível

Orin, FP32, batch de 8 clipes de ajuste, 2 aquecimentos e 5 medições:

| Plano | ms/clipe | p50 | p95 | Speedup medido | Previsto (`budget_cost`) |
|---|---:|---:|---:|---:|---:|
| Denso | 2199 | 2205 | 2207 | 1,00× | 1,00× |
| 512 | 579 | 580 | 580 | 3,80× | 3,93× |
| 1024 | 994 | 994 | 995 | 2,21× | 2,08× |
| 1536 | 1451 | 1523 | 1530 | 1,52× | 1,37× |
| 2048 (caminho denso) | 1871 | 1871 | 1872 | 1,18× | 1,00× |

O nível de 2048 faz a mesma conta que o denso, mas foi medido 15% mais rápido. Ou seja, o
denso foi cronometrado num estado mais lento do Orin: ele foi o primeiro plano, e os clocks
ou a temperatura podem ter pesado. Speedups vindos de uma única medição sequencial carregam
essa incerteza, de cerca de 15% aqui. No benchmark final, é melhor alternar e repetir as
variantes.

### O que o P0 permite concluir — e o que não

**Permite:**
- O caminho podado está correto (G1).
- Com o scoring, o merge e a atenção atuais, cortar após o bloco 2 afasta muito o embedding
  final, mesmo com 75% dos tokens. Nenhuma política de orçamento chega a cosseno ≥ 0,90 com
  economia relevante sem mudar o mecanismo: em 1536 tokens, só 23% dos clipes chegam lá.
- O desvio não é explicado pelo erro local do merge; ele cresce nos blocos profundos.
- O probe, como desenhado, não mostrou sinal neste painel.

**Não permite:**
- Concluir que a QABP é inviável. G2 é um gate diagnóstico, com n = 26.
- Afirmar que atenção proporcional ou um corte mais tardio resolvem o problema; isso não foi
  testado.
- Dizer qualquer coisa sobre o AMA ou o mAP.
