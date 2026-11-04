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

## 2. Validação downstream reduzida — conjunto de seleção

Depois do P0, o G2 passou a ser diagnóstico, e o critério principal passou a ser a tarefa:
o AMA congelado reconhecendo e localizando ações contra as anotações, com custo real
medido.

- **Conjunto de seleção:** motoristas 28557 e 30932 (4 vídeos), separados do ajuste da QABP
  (13522, 31903, 33682), do holdout (35418, 36305) e da avaliação final (38159, 38479).
- **Condições:** as mesmas do P0 (FP32, batch 8, corte após o bloco 2). Sem prior, exceto
  onde a seção diz que usa a máscara do motorista.
- **Anotações:** `label_A1-train_A1-val.json`.
- **Ressalva:** os 16 vídeos locais estão no subset de **treino** do AMA, e o AMA foi
  treinado com recortes do motorista. O mAP mede o quanto cada variante preserva o
  desempenho do pipeline, não a generalização.

### 2.1 Atenção proporcional no painel do P0 (26 clipes; drift só como sinal auxiliar)

`bash benchmark/run_qabp.sh atencao`: o mesmo painel e a mesma seleção, com
`size_biased_attention`.

| Tokens | Atenção | Cosseno P10 / P50 / P90 | Top-1 igual ao denso | Cosseno após o bloco 20 → 40 |
|---:|---|---:|---:|---:|
| 1024 | padrão | 0,281 / 0,522 / 0,828 | 42% | 0,859 → 0,550 |
| 1024 | proporcional | 0,236 / 0,571 / 0,835 | 42% | 0,877 → 0,582 |
| 1536 | padrão | 0,557 / 0,739 / 0,959 | 69% | 0,958 → 0,731 |
| 1536 | proporcional | 0,548 / 0,779 / 0,967 | 65% | 0,964 → 0,768 |

A atenção proporcional sobe a mediana do cosseno em 0,04–0,05, mas não melhora o P10 nem a
concordância top-1, e o desvio continua nascendo depois do bloco 10. Pelo protocolo, isso
não basta para adotá-la; ela passa pelo AMA na rodada de seleção.

### 2.2 Conferência do denso (`28557_NoAudio_5`: 914 clipes, 16 ações anotadas)

`bash benchmark/run_qabp.sh sanidade`. O critério era parar se o mAP médio do denso ficasse
abaixo de 0,05; não ficou.

| | Denso | Fixo 1536 |
|---|---:|---:|
| mAP@0,1 a @0,5 e médio | 1,00 | 1,00 |
| Ações reencontradas (tIoU ≥ 0,5, confiança ≥ 0,3) | 16/16 | 15/16 |
| Precisão contra as anotações / falsos positivos | 0,89 / 2 | 0,88 / 2 |
| Erro médio da fronteira, início / fim (s) | 0,57 / 1,22 | 0,70 / 1,09 (denso nas mesmas 15 ações: 0,57 / 1,26) |
| Classe da cabeça igual à anotada, por clipe / por ação | 82% / 88% | 82% / 88% |
| Classe da cabeça igual à do denso, por clipe | 100% | 64% |
| Cosseno com o denso, P10 / P50 | 1 / 1 | 0,63 / 0,78 |
| Tempo por clipe (média, p95) | 1,90 s, 2,00 s | 1,33 s, 1,39 s |
| Speedup medido (previsto) | 1,00× | 1,43× (1,37×) |

**Leitura:**
- Com 1536 tokens, a qualidade contra as anotações praticamente não muda, embora o embedding
  se afaste do denso e a cabeça mude de classe em 36% dos clipes. A concordância com o denso,
  e o cosseno, não medem a perda na tarefa: a cabeça acerta as anotações tanto quanto o
  denso.
- **Teto:** o denso acerta todas as ações, e as fronteiras ficam a cerca de 1 s das
  anotadas. É o comportamento esperado de um modelo avaliado num vídeo do próprio treino.
  O mAP está saturado, e degradações pequenas ficam invisíveis. Sinais mais finos (ações
  reencontradas com confiança ≥ 0,3, fronteiras, confiança das detecções) ajudam, mas a
  avaliação independente precisa de vídeos fora do treino do AMA: os 18 vídeos Dashboard
  do subset `validation`, que não estão nesta máquina.
- Na primeira tentativa, o avaliador do AMA falhou: ele usa `np.float`, removido do NumPy
  1.24 (aqui roda a 2.2.6). A correção recria esse alias (o `float` nativo) e roda o
  avaliador num único processo; o código do AMA não foi tocado (`8505d29`).

### 2.3 Rodada de seleção — interrompida após o 1º vídeo

`bash benchmark/run_qabp.sh selecao`, com 4 vídeos e as variantes denso, fixo
1536/1024/512 e fixo 1536/1024 com atenção proporcional (AP). A ordem das variantes alterna
a cada batch.

A rodada parou no 2º vídeo, quando o Jetson reiniciou sozinho. Ele reiniciou de novo uma
hora depois, outra vez com a GPU em carga contínua. A causa é desconhecida: o journal do
sistema não é persistente. Numa rodada posterior de 1h15, o `tegrastats`
(`logs/tegrastats.log`) registrou no máximo 71 °C e 43 W, longe dos limites do Orin.

Só o vídeo `28557_NoAudio_5` ficou completo (914 clipes, 16 ações, uma de cada classe).
Ele foi avaliado à parte em `selecao-parcial/28557_NoAudio_5/`. Fronteiras e confiança
aparecem como "variante / denso nas mesmas ações", sobre as ações que os dois encontram
(ver §3).

| | Denso | 1536 | 1024 | 512 | 1536 + AP | 1024 + AP |
|---|---:|---:|---:|---:|---:|---:|
| Ações reencontradas | 16/16 | 15/16 | 9/16 | 4/16 | 15/16 | 9/16 |
| mAP médio | 1,00 | 1,00 | 0,93 | 0,53 | 1,00 | 0,90 |
| Falsos positivos | 2 | 2 | 6 | 8 | 3 | 3 |
| Início (s) | 0,57 | 0,70 / 0,57 | 0,55 / 0,37 | 0,67 / 0,36 | 0,58 / 0,57 | 0,68 / 0,37 |
| Fim (s) | 1,22 | 1,09 / 1,26 | 1,64 / 0,78 | 3,17 / 0,20 | 0,97 / 1,26 | 1,80 / 0,78 |
| Confiança | 0,79 | 0,77 / 0,82 | 0,76 / 0,85 | 0,59 / 0,82 | 0,78 / 0,82 | 0,73 / 0,85 |
| Cosseno P50 com o denso | 1 | 0,78 | 0,52 | 0,08 | 0,79 | 0,49 |
| Tempo por clipe | 2,05 s | 1,44 s | 0,94 s | 0,55 s | 1,36 s | 0,91 s |
| Speedup medido | 1,00× | 1,43× | 2,19× | 3,75× | 1,51× | 2,25× |

**Leitura:**
- **512 quebra a inferência:** só 4 ações e 8 falsos positivos.
- **1536 fica praticamente igual ao denso.**
- **1024 perde 7 ações, todas sutis:** bocejar, cantar, conversar com o banco de trás,
  digitar à direita, ajustar o painel, pegar algo do chão do lado do passageiro e a direção
  normal. Os gestos grandes sobrevivem: beber, comer, pegar algo do chão do lado do
  motorista e alcançar o banco de trás são encontrados até em 512.
- **O mAP satura:** com uma ação por classe, a contagem de ações reencontradas é mais
  informativa que ele.
- **A atenção proporcional não ganha na tarefa:** empata no 1536 e, no 1024, encontra as
  mesmas 9 ações com menos falsos positivos. Ela saiu 3–5% mais rápida porque usa o kernel
  de atenção do PyTorch (`scaled_dot_product_attention`), enquanto a atenção do modelo
  monta a matriz `q·kᵀ` inteira. Trocar esse kernel pode acelerar todas as variantes; isso
  não foi explorado.

### 2.4 Por que um teste antigo com 512 tokens tinha sido favorável

O teste antigo (`outputs-temp`, setembro de 2026, em CPU) usou 512 tokens com corte após o
bloco 2. Ele rodou num trecho de 19 s **recortado no motorista** (`28558_NoAudio_5`,
`crop_videos`) e centrado numa ação de beber. O AMA encontrou a mesma ação, e o cosseno
P50 com o denso foi 0,955 (P10 0,51).

Conferi que o algoritmo, o pré-processamento e os pesos são os mesmos de hoje. O código
de merge, scoring, atenção e encoder é idêntico, e a `LayerNorm` final do checkpoint
reproduz exatamente os descritores antigos.

Para separar recorte e conteúdo, `benchmark/qabp_crop.py` mediu o cosseno com o denso em
512 tokens, na CPU:

| Clipe | Conteúdo | Quadro inteiro | Recorte |
|---|---|---:|---:|
| `13522_7` ~201 s | Drinking | 0,75 | 0,97 |
| `30932_5` ~360 s | Drinking | 0,98 | 0,99 |
| `13522_7` ~391 s | Reaching behind | 0,04 | 0,15 |
| `13522_7@0` | sem ação anotada | −0,15 | 0,31 |
| `30932_5@0` | sem ação anotada | 0,12 | 0,18 |
| `31903_5@0` | sem ação anotada | 0,79 | 0,83 |

A CPU reproduz a GPU do P0 com diferença de até 0,002. **O conteúdo pesa mais que o
recorte.** Gestos localizados, como beber, sobrevivem à poda, e os demais clipes desabam.
O recorte ajuda pouco. O resultado se decide nos 10 últimos blocos: até o bloco 30 todos
os clipes se afastam do denso de forma parecida (~0,5–0,6) e, depois, uns voltam à
resposta do denso e outros desabam.

### 2.5 Quais tokens × quantos tokens (vídeo `28557_NoAudio_5`)

**Para onde vão os tokens** (`selecao-tokens/`: um clipe no meio de cada uma de 10 ações,
só até o bloco 2, sem máscara):
- **Em 512:** 30 das 64 posições de cada fatia, ou 44% dos tokens, são as mesmas em pelo
  menos 9 dos 10 clipes, seja qual for a ação. São a luz do teto, as bordas das janelas, os
  encostos dos bancos de trás e o carimbo de data e hora queimado no vídeo. O carimbo
  sozinho fica com 7% dos tokens e ocupa 2% da imagem.
- **Em 1024:** 112 das 128 posições se repetem da mesma forma.
- **Nas ações sutis,** o rosto e as mãos recebem poucos tokens.

**Mesma quantidade, critérios diferentes** (`benchmark/qabp_selection.py`: `criterio-512/`
e `criterio-1024/`, mais `selecao` para o critério atual). Os números são ações
reencontradas e, entre parênteses, o mAP médio:

| Critério | 512 | 1024 |
|---|---:|---:|
| Atual (movimento + unicidade, sem máscara) | 4/16 (0,53) | 9/16 (0,93) |
| Aleatório | 3/16 (0,30) | 14/16 (0,94) |
| Máscara do motorista\* | 5/16 (0,77) | 16/16 (1,00) |
| Máscara\* sem o carimbo de data | 5/16 (0,55) | 16/16 (1,00) |

\* `region-novo.json`. Essa máscara foi calibrada nos 16 vídeos locais, este incluído, e
por isso tem vazamento; o §2.6 a substitui pela máscara justa. O braço "só movimento"
(`motion_weight` = 1) foi descartado: ele mantinha os tokens da 1ª fatia pela posição na
grade (corrigido em `518c321`, ver §3).

**Leitura:**
- **Em 512, a quantidade domina:** nenhum critério passa de 5/16.
- **Em 1024, o critério decide.** O atual é pior que o aleatório, porque gasta o orçamento
  em pontos fixos da cena. Com a máscara, as 16 ações voltam.

### 2.6 Máscara justa: só os motoristas de ajuste

`benchmark/results/qabp/regiao/regiao-ajuste.json` é a mediana, por coordenada, das
maiores caixas do motorista nos vídeos de ajuste (13522, 31903, 33682). As caixas vêm do
próprio `region-novo.json`, e o arquivo registra essa origem. O resultado é
[0,099; 0,207; 0,850; 0,994], com 169 posições contra 156; a diferença é uma linha a mais
no alto. Nos estágios em que a região escolhe tokens, o `run_qabp.sh` usa essa máscara
(`MASK`).

Vídeo `28557_NoAudio_5` em 1024 tokens (`criterio-1024-mascara-justa/`):
- **Resultado:** 15/16 ações, mAP 1,00 e 2 falsos positivos, como o denso.
- **A 16ª ação foi encontrada por um fio.** "Adjust control panel" (467–478 s) saiu na
  classe certa, em 465,4–476,3 s (tIoU 0,74), com confiança 0,299, logo abaixo do
  limiar de 0,3.
- **Nas 15 ações em comum com o denso:** início 0,57 s contra 0,58 s, fim 1,19 s contra
  1,08 s e confiança 0,72 contra 0,79.
- **Custo:** 0,94 s por clipe (2,18×).

### 2.7 Confirmação nos outros vídeos de seleção

`bash benchmark/run_qabp.sh confirmacao` (`confirmacao/`): vídeos `28557_7`, `30932_5` e
`30932_7` (48 ações), máscara justa, `benchmark run --masked 1024 768`. Foram ~2h50 de GPU,
sem queda.

| | Denso | 1024 + máscara | 768 + máscara |
|---|---:|---:|---:|
| Ações reencontradas | 48/48 | 41/48 | 36/48 |
| mAP médio | 1,00 | 0,95 | 0,89 |
| Falsos positivos | 3 | 5 | 10 |
| Início (s): variante / denso nas mesmas ações | 0,55 | 0,68 / 0,44 | 0,76 / 0,43 |
| Fim (s): variante / denso nas mesmas ações | 0,65 | 0,84 / 0,52 | 0,85 / 0,53 |
| Confiança: variante / denso nas mesmas ações | 0,80 | 0,76 / 0,82 | 0,71 / 0,82 |
| Tempo por clipe | 2,03 s | 0,94 s | 0,78 s |
| Speedup medido | 1,0× | 2,17× | 2,60× |

Ações reencontradas por vídeo (o denso acerta 16/16 em todos):

| Vídeo | 1024 + máscara | 768 + máscara |
|---|---:|---:|
| `28557_7` | 15/16 | 10/16 |
| `30932_5` | 14/16 | 13/16 |
| `30932_7` | 12/16 | 13/16 |

O 1024 com máscara perdeu 7 ações:
- **5 perdas de verdade, todas do motorista 30932:** direção normal (nos dois vídeos),
  conversar com o banco de trás, ajustar o painel e bocejar.
- **2 por fronteira:** Phone R (tIoU 0,39; o próprio denso fica em 0,54) e Eating (0,31).

**Leitura:**
- **A máscara é a mudança certa,** mas o 1024 com máscara não equivale ao denso em todos os
  motoristas. O 28557 fica em 15/16 nos dois vídeos; o 30932 perde de 2 a 4 ações sutis por
  vídeo.
- **Fronteiras:** nas ações em comum com o denso, ficam cerca de 55–60% mais imprecisas.
- **O 768 não compensa:** é só ~17% mais rápido que o 1024, perde mais ações e dobra os
  falsos positivos.
- **Em aberto:**
  - o 1536 com máscara no motorista 30932;
  - a perda aceitável (δ), que ainda não foi definida.

## 3. Correções de código desta etapa (auditoria do repositório)

| Commit | Correção | Efeito nos resultados |
|---|---|---|
| `a8d375f` | O relatório compara fronteiras e confiança só nas ações que a variante e o denso encontram, e mostra o denso nessas mesmas ações | Mudou a leitura: no §2.7, as fronteiras do 1024 com máscara pareciam 0,13 / 0,19 s piores que as do denso; nas mesmas ações, são 0,24 / 0,33 s piores |
| `518c321` | `motion_weight` = 1 é recusado | O braço "só movimento" do §2.5 era inválido e foi descartado |
| `e9d8017` | `benchmark run` grava as linhas antes das features e cada arquivo de features inteiro, e a retomada descarta o que uma queda deixou e refaz o vídeo | Proteção contra as reinicializações do Jetson |
| `94669c6` | `budget-curve` refaz um vídeo que o `sources.json` não lista e grava esse arquivo inteiro | Uma queda não tira mais um vídeo da calibração sem aviso |
| `a8da0af` | O `run_qabp.sh` só pula o AMA quando o manifesto final existe, e o relatório avisa quais variantes ficaram sem saída do AMA | Um AMA interrompido não gera mais um relatório sem métricas sem aviso |
| `dc8583b` | Uma região calibrada em vídeos ou motoristas avaliados é recusada onde escolhe tokens; a sobreposição vai para o `run.json`; o script de critérios grava a região de fato usada | O `region-novo.json` não serve mais para escolher tokens; o estudo de máscaras antigo, calibrado nos vídeos avaliados, não roda mais com ele |

Também entraram nesta etapa:
- `benchmark run --masked`, para orçamentos fixos com a máscara do motorista (`7cbc219`);
- os diagnósticos `benchmark/qabp_crop.py` e `benchmark/qabp_selection.py`;
- os estágios `proxy`, `criterio` e `confirmacao` do `run_qabp.sh`.

## 4. Estado e pendências

- **`selecao`:** só o 1º vídeo está completo, e os outros 3 não têm 1536 nem atenção
  proporcional; a confirmação do §2.7 cobre o 1024 e o 768 com máscara.
  `bash benchmark/run_qabp.sh selecao` retoma a partir do 2º vídeo. Nesse estudo de
  políticas, o `region-novo.json` continua só como região registrada, sem escolher tokens.
- **Adiadas pelo dono:**
  - a Fase B (estudo de proxies);
  - a Fase C (QABP orientada à tarefa).

  Qualquer política adaptativa, legada ou QABP, precisa ser recalibrada com a máscara:
  a QABP v1 recusa máscaras.
- **Limites que continuam valendo:**
  - os vídeos locais estão no subset de treino do AMA;
  - o benchmark usa o quadro inteiro, e o AMA foi treinado com recortes;
  - os 18 vídeos de validação independente não estão nesta máquina;
  - a causa das reinicializações do Jetson é desconhecida.
