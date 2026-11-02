# Ficha do experimento

Condições em que os resultados de `results/` foram produzidos. Os campos _(a preencher)_
ficam por conta de quem roda o experimento. O `run.json`, o `policy.json`, o
`region.json` e a aba **Configuração** do relatório já registram sozinhos o ambiente da
execução (Python, torch, GPU), o SHA-256 dos pesos e do detector, e a identidade da
política e da região.

## Perguntas

1. A máscara inteligente (região do motorista) preserva mais que a máscara fixa?
2. O ViT-Prune mantém o resultado do denso na ação inteira?
3. Quantos tokens ficam após a poda, onde ficam e quanto isso economiza?

## Dados

| Item | Valor |
|---|---|
| Base | AI City Challenge 2024, Track 3, câmera Dashboard _(confirmar a partição)_ |
| Vídeos de calibração | 16 _(listar)_ |
| Trechos por vídeo na calibração | 160, espaçados pelo vídeo |
| Vídeos avaliados | `Dashboard_user_id_13522_NoAudio_5`, `Dashboard_user_id_13522_NoAudio_7` |
| Anotação | `videos/class-intervals.json`: 16 classes, 16 ações por vídeo |
| Entrada do modelo | quadro inteiro (1920x1080) redimensionado para 224x224 |
| Clipe | 16 quadros com passo 2 (32 quadros, cerca de 1 s), um a cada 16 quadros |
| Duração total avaliada | _(a preencher)_ |

## Modelos

| Modelo | Treino de origem | Arquivo |
|---|---|---|
| VideoMAEv2-g (ViT-g/14) | ajuste fino AI City Track 3, recortes do motorista _(confirmar dados)_ | `videomae-v2_finetune_aicity.pth` |
| AMA | solução AI City, features VideoMAE _(confirmar dados)_ | `best_performance.pth.tar`, `aicity_ego_vitl.yaml` |
| Detector | YOLO26s pré-treinado no COCO (Ultralytics) | `yolo26s.pt` |

Nenhum modelo é retreinado: o ViT-Prune só calibra a política e a região.

## ViT-Prune

| Parâmetro | Valor |
|---|---|
| Camada de poda | depois do bloco 2 de 40 |
| Níveis de tokens por clipe | 256, 512 e 768 de 2048 |
| Peso do movimento na política | 0,5 |
| Máscara fixa | 4 linhas do topo com peso 0,15 |
| Máscara inteligente | maior pessoa com confiança acima de 0,5 no quadro do meio de cada trecho de calibração, unida na grade 16x16 |

## Ambiente

| Item | Valor |
|---|---|
| Dispositivo alvo (onde o modelo vai rodar) | _(a preencher)_ |
| Máquina da medição: GPU, driver, CUDA | _(a preencher)_ |
| CPU e memória | _(a preencher)_ |
| Sistema operacional | _(a preencher)_ |
| Python, torch, torchvision | _(a preencher)_ |
| timm, ultralytics, opencv-python, numpy | _(a preencher)_ |
| Commit do vitprune (branch `benchmark`) | _(a preencher)_ |
| Commit do AMA | _(a preencher)_ |
| Precisão e lote | `precision` e `batch` do `settings.json` |

`pip freeze > results/pip-freeze.txt`, no ambiente usado, guarda todas as versões.

## Como reproduzir

Na raiz do repositório, num ambiente com `pip install -e ".[pipeline,region]"` e
`pip install -r benchmark/requirements.txt`. Os vídeos de calibração ficam numa pasta
sem os vídeos avaliados. O `yolo26s.pt` vem de
<https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt>.

```powershell
vitprune calibrate --videos <calibracao> --weights <videomae.pth> --model-source <feature_extraction> --levels 256 512 768 --clips-per-video 160 --precision off --device cuda --out benchmark/results/policy.json
vitprune region --videos <calibracao> --detector yolo26s.pt --clips-per-video 160 --device cuda --out benchmark/results/region.json
python benchmark/run_inference.py
python benchmark/evaluate_ama.py
python benchmark/make_report.py
```

`--precision` do `calibrate` deve ser o mesmo `precision` do `settings.json`.
