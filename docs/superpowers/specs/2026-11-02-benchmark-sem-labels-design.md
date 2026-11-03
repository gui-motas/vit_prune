# Benchmark sem labels: o pipeline denso como referência

## Intenção

Os vídeos de teste não têm anotação, então o benchmark atual, que lê o
`class-intervals.json` em três lugares, não roda. A pergunta do benchmark não pede
anotação: **o ViT-Prune mantém as respostas do pipeline enquanto reduz o tempo por
clipe?** A referência passa a ser a saída do próprio pipeline denso, em dois níveis: a
cabeça de classificação do VideoMAE e as detecções do AMA.

## Objetivos

- `vitprune benchmark run | ama | report` funcionam sem nenhum arquivo de anotação.
- As 4 variantes continuam (`denso`, `vitprune`, `vitprune_mascara_fixa`,
  `vitprune_mascara_inteligente`), com o tempo por clipe, os tokens mantidos, a parcela
  dentro da região do motorista e o cosseno com o denso, como hoje.
- Cada variante é comparada com o denso na cabeça e nas detecções do AMA.

## Fora do escopo

- Acurácia contra a verdade, mAP em qualquer forma e nomes de classe.
- Alterar o AMA ou usar o `eval.py` dele.
- Ler o vídeo por "ação anotada": sem labels não há ações.

## O que não pode acontecer

Um número que pareça acurácia e seja concordância. O relatório diz "concordância com o
denso", nunca "acerto". Se o denso errar, uma variante que erra igual concorda; a medida
mostra o quanto a poda muda o resultado, não se ele está certo.

## Restrições

- O AMA fica como está, rodando no interpretador dele (loguru, torch), com o
  `nms_1d_cpu` em Python no lugar da extensão compilada.
- O formato das features (`feats`, `a_prob`) não muda.
- Uma variante já avaliada pelo AMA é recusada, como hoje.

## Abordagens consideradas

1. **`eval.py` com um arquivo de anotações de fachada.** Funciona sem alterar o AMA: o
   avaliador grava o `predictions.csv` antes de calcular o mAP. Mas mantém um avaliador
   que existe para comparar com uma verdade que não temos, exige um arquivo de 16
   classes e descarta o mAP que imprime. Descartada.
2. **mAP de cada variante contra as detecções do denso.** Exige o avaliador e uma
   passada extra, o denso sai com 100% por construção e o valor depende do limiar.
   Descartada.
3. **Rodar o modelo do AMA direto sobre as features (escolhida).** Testei com os pesos
   reais e features aleatórias: todas as chaves casam e o modelo devolve segmentos em
   segundos, confiança e classe. Precisa só de `feats`, `fps`, `duration`, `feat_stride`,
   `feat_num_frames` e `video_id`.

## Desenho

**`benchmark run`.** Sai o `--annotations`. Os clipes deixam de pertencer a uma ação: o
`plan` só lista os clipes, e o `clips.jsonl` perde o campo `segment`. As figuras usam
`--show N` clipes espalhados por igual em cada vídeo (padrão 6, escolha minha). O
`run.json` registra, por vídeo, o fps, os quadros e os clipes, para o AMA e o relatório.

**`benchmark ama`.** Um script autocontido, dentro do pacote (só as bibliotecas do AMA,
numpy e torch; não importa o vitprune), roda como subprocesso com o diretório do AMA como
`cwd` e o `ama_compat` no `PYTHONPATH`. Carrega a configuração e os pesos uma vez, como o
`eval.py` (DataParallel, `state_dict_ema`, semente 0), e para cada variante e vídeo monta
a entrada a partir do `.npz` e grava `ama/<variante>/predictions.csv` (`video-id, t-start,
t-end, label, score`), o `downstream.log` e um manifesto com os hashes da configuração,
dos pesos e das features. Passo, `num_frames` e fps vêm das features e da configuração do
AMA; a duração é quadros/fps. Sem `ama.json`. Flag nova: `--ama-python` (padrão: o
interpretador atual), para o vitprune e o AMA viverem em ambientes diferentes.

**`benchmark report`.** Referência = o denso.

- *Cabeça:* por clipe, a parcela em que a classe de maior probabilidade da variante é a do
  denso, e o cosseno (já existe).
- *AMA:* as detecções do denso com confiança ≥ `--min-score` (0,3) são a referência. Por
  variante: **recall** (parcela delas que tem, na variante, uma detecção da mesma classe
  com tIoU ≥ `--tiou`, padrão 0,5), **precisão** (parcela das detecções da variante, também
  ≥ `--min-score`, que casam com uma do denso) e o **tIoU médio** do melhor par. O denso
  é a referência: 100% por definição, marcado como tal.
- *Abas:* Resumo, **Por detecção do denso** (no lugar de "Por ação": classe, janela e, por
  variante, o melhor tIoU e a classe que a cabeça dá nos clipes dentro da janela), Por
  vídeo e Configuração. Saem "Por ação" e "AMA (mAP)".
- *Figuras:* máscaras e seleção de tokens como hoje, nos clipes espalhados; a linha do
  tempo troca "Anotação real" por "Denso (referência)"; o gráfico-resumo troca mAP e
  "ações reconhecidas" por recall e concordância de classe. Classes aparecem como
  "classe N".

## Testes

- Os do relatório passam a usar detecções do denso no lugar de anotações.
- Um projeto AMA falso (pacote `libs` mínimo em tmp) exercita o script do AMA de ponta a
  ponta, sem GPU nem pesos.
- O `run` continua coberto pelo backbone em miniatura e vídeo sintético.
- Conferi o script direto com o AMA real uma vez, à mão.

## Fica para quem construir

Nomes exatos das colunas, o formato do manifesto do AMA e como casar detecções quando
várias da variante cobrem a mesma da referência (o melhor tIoU de cada uma basta).
