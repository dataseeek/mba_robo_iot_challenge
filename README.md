# Detecção de Pessoas — RoboIoT

Detecção binária de pessoas (PESSOA / NENHUMA) usando Transfer Learning com MobileNet, com servidor de inferência FastAPI projetado para implantação em hardware modesto (ex: Raspberry Pi).

Baseado na técnica de Transfer Learning adaptada para detecção binária de pessoas com quantização TFLite INT8 (modelo de 155 KB).

## Arquitetura

```
                                  MobileNet (alpha=0.25, pesos ImageNet)
                                  Corte em conv_pw_10_relu
                                           |
                                     Reshape → Dropout(0.1) → Flatten
                                           |
                                   Dense(2, softmax)
                                    /            \
                              NENHUMA(0)      PESSOA(1)
```

- **Entrada**: Imagem RGB 96x96x3
- **Saída**: Classificação binária com score de confiança
- **Tamanho do modelo**: ~155 KB (TFLite quantizado INT8)
- **Treinamento**: 80 epochs base congelada + 1 epoch fine-tuning (LR=0.00001)
- **Acurácia**: ~95% no subconjunto COCO val2017

## Estrutura do Projeto

```
mba_robo_iot_challenge/
├── config.py                  # Constantes compartilhadas (arquitetura, hiperparâmetros, limiares)
├── download_dataset.py        # Download do dataset COCO pessoa/sem-pessoa
├── train.py                   # CLI de treinamento (requer TensorFlow completo)
├── api.py                     # Servidor de inferência FastAPI (apenas TFLite runtime)
├── stream_client.py           # Cliente webcam para detecção remota
├── requirements.txt           # Dependências
├── person-detect-train.ipynb  # Notebook de treinamento (detecção de pessoas)
├── data/                      # Dados de treinamento (baixar com download_dataset.py)
│   ├── NENHUM/                #   Imagens sem pessoa
│   └── PESSOA/                #   Imagens com pessoa
└── model/                     # Modelos treinados
    ├── person-detect-model.tflite       # Float16 (~217 KB)
    └── person-detect-model-int8.tflite  # INT8 quantizado (~155 KB, deploy)
```

## Início Rápido

### 1. Configuração

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow   # para treinamento
pip install fastapi uvicorn[standard] python-multipart opencv-python-headless numpy pydantic websockets
```

### 2. Download do Dataset

Baixa um subconjunto balanceado do COCO 2017 (imagens com pessoa vs sem pessoa):

```bash
python download_dataset.py --count 1000 --split val --workers 8
```

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--count` | 1000 | Imagens por classe |
| `--split` | val | Split do COCO (`val` = 5K imagens, `train` = 118K) |
| `--workers` | 4 | Threads de download paralelo |
| `--cache` | .coco_cache | Diretório de cache das anotações |

### 3. Treinamento

```bash
python train.py --dataset ./data --output ./model
```

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--dataset` | *(obrigatório)* | Caminho do dataset com subdiretórios NENHUM/ e PESSOA/ |
| `--output` | ./output | Diretório de saída dos modelos |
| `--frozen-epochs` | 40 | Epochs da fase 1 (base MobileNet congelada) |
| `--finetune-epochs` | 20 | Epochs da fase 2 (fine-tuning do modelo completo) |
| `--batch-size` | 100 | Tamanho do batch de treinamento |
| `--no-finetune` | - | Pular fase de fine-tuning |
| `--no-quantize` | - | Pular exportação TFLite INT8 |

Teste rápido:

```bash
python train.py --dataset ./data --output ./model --frozen-epochs 2 --finetune-epochs 1
```

### 4. Executar API

```bash
python api.py --model ./model/person-detect-model.tflite
```

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--model` | auto-detectar | Caminho do modelo .tflite |
| `--host` | 0.0.0.0 | Endereço de bind |
| `--port` | 8000 | Porta |
| `--confidence` | 0.6 | Limiar de confiança para detecção |
| `--fps` | 5 | FPS do stream de câmera |

Ou via uvicorn:

```bash
MODEL_PATH=./model/person-detect-model.tflite uvicorn api:app --host 0.0.0.0 --port 8000
```

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Interface Web — arraste e solte imagens para teste visual |
| `GET` | `/health` | Health check + status do modelo |
| `POST` | `/detect` | Upload de imagem → resultado JSON |
| `POST` | `/detect/image` | Upload de imagem → JPEG anotado com overlay de detecção |
| `POST` | `/stream/start` | Iniciar processamento de stream câmera/RTSP |
| `POST` | `/stream/stop` | Parar stream de câmera |
| `GET` | `/stream/status` | Estado atual do stream + último resultado |
| `WS` | `/ws/stream` | Resultados de detecção em tempo real via WebSocket |

### Exemplos

**Detectar (JSON)**:
```bash
curl -X POST -F "file=@foto.jpg" http://localhost:8000/detect
```
```json
{
  "detected": true,
  "class": "PESSOA",
  "confidence": 1.0,
  "raw_scores": {"NENHUMA": 0.0, "PESSOA": 1.0}
}
```

**Detectar (imagem anotada)**:
```bash
curl -X POST -F "file=@foto.jpg" http://localhost:8000/detect/image -o resultado.jpg
```

**Iniciar stream de câmera**:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"source": 0}' http://localhost:8000/stream/start
```

**Stream RTSP**:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"source": "rtsp://usuario:senha@ip-camera:554/stream"}' http://localhost:8000/stream/start
```

## Cliente de Stream Webcam

Para detecção remota via webcam na rede local. Execute em qualquer máquina com webcam:

```bash
pip install opencv-python requests
python stream_client.py --api http://<ip-servidor>:8000
```

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--api` | *(obrigatório)* | URL da API (ex: `http://<ip-servidor>:8000`) |
| `--camera` | 0 | Índice da webcam |
| `--delay` | 1.0 | Segundos entre detecções |

Abre uma janela mostrando o feed anotado ao vivo. Pressione `q` para sair.

## Deploy no Raspberry Pi

No RPi, instale apenas as dependências leves de inferência (sem TensorFlow completo):

```bash
pip install tflite-runtime
pip install fastapi uvicorn[standard] python-multipart opencv-python-headless numpy pydantic websockets
```

Copie apenas estes arquivos para o RPi:
- `config.py`
- `api.py`
- `model/person-detect-model.tflite`

```bash
python api.py --model person-detect-model.tflite --port 8000
```

## Stack Tecnológico

| Componente | Tecnologia |
|------------|------------|
| Treinamento | TensorFlow/Keras 3, MobileNet |
| Inferência | TFLite Runtime (quantizado INT8) |
| API | FastAPI, Uvicorn |
| Processamento de imagem | OpenCV |
| Dataset | COCO 2017 |
| Tempo real | WebSocket, threading |
