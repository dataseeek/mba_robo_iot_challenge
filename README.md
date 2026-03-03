# Person Detection API — RoboIoT

Binary person detection (PESSOA / NENHUMA) using Transfer Learning on MobileNet, with a FastAPI inference server designed for deployment on modest hardware (e.g. Raspberry Pi).

Based on the Transfer Learning technique from `model-train.ipynb`, adapted for binary person detection with INT8 TFLite quantization (155 KB model).

## Architecture

```
                                  MobileNet (alpha=0.25, ImageNet weights)
                                  Cut at conv_pw_10_relu
                                           |
                                     Reshape → Dropout(0.1) → Flatten
                                           |
                                   Dense(2, softmax)
                                    /            \
                              NENHUMA(0)      PESSOA(1)
```

- **Input**: 96x96x3 RGB image
- **Output**: Binary classification with confidence score
- **Model size**: ~155 KB (INT8 quantized TFLite)
- **Training**: 40 frozen epochs + 1 fine-tune epoch (LR=0.00001)
- **Accuracy**: ~95% on COCO val2017 subset

## Project Structure

```
roboiot/
├── config.py              # Shared constants (architecture, hyperparameters, thresholds)
├── download_dataset.py    # Download COCO person/no-person dataset
├── train.py               # Training CLI (requires full TensorFlow)
├── api.py                 # FastAPI inference server (TFLite runtime only)
├── stream_client.py       # Webcam client for remote detection
├── requirements.txt       # Dependencies
├── model-train.ipynb      # Reference notebook (cat classification)
├── dataset/               # Training data (user-provided or downloaded)
│   ├── NENHUM/            #   Images with no person
│   └── PESSOA/            #   Images with person
└── output/                # Generated models
    ├── person-detect-model.keras    # Full Keras model
    └── person-detect-model.tflite   # INT8 quantized (deploy this)
```

## Quick Start

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow   # for training
pip install fastapi uvicorn[standard] python-multipart opencv-python-headless numpy pydantic websockets
```

### 2. Download Dataset

Downloads a balanced subset from COCO 2017 (person vs no-person images):

```bash
python download_dataset.py --count 1000 --split val --workers 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | 1000 | Images per class |
| `--split` | val | COCO split (`val` = 5K images, `train` = 118K) |
| `--workers` | 4 | Parallel download threads |
| `--cache` | .coco_cache | Annotation cache directory |

### 3. Train

```bash
python train.py --dataset ./dataset --output ./output
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | Path to dataset with NENHUM/ and PESSOA/ subdirs |
| `--output` | ./output | Output directory for models |
| `--frozen-epochs` | 40 | Phase 1 epochs (frozen MobileNet base) |
| `--finetune-epochs` | 20 | Phase 2 epochs (full model fine-tuning) |
| `--batch-size` | 100 | Training batch size |
| `--no-finetune` | - | Skip fine-tuning phase |
| `--no-quantize` | - | Skip TFLite INT8 export |

Quick sanity check:

```bash
python train.py --dataset ./dataset --output ./output --frozen-epochs 2 --finetune-epochs 1
```

### 4. Run API

```bash
python api.py --model ./output/person-detect-model.tflite
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | auto-detect | Path to .tflite model |
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Port |
| `--confidence` | 0.6 | Detection confidence threshold |
| `--fps` | 5 | Camera stream FPS |

Or via uvicorn:

```bash
MODEL_PATH=./output/person-detect-model.tflite uvicorn api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI — drag & drop images for visual testing |
| `GET` | `/health` | Health check + model status |
| `POST` | `/detect` | Upload image → JSON result |
| `POST` | `/detect/image` | Upload image → annotated JPEG with detection overlay |
| `POST` | `/stream/start` | Start camera/RTSP stream processing |
| `POST` | `/stream/stop` | Stop camera stream |
| `GET` | `/stream/status` | Current stream state + latest result |
| `WS` | `/ws/stream` | Real-time detection results via WebSocket |

### Examples

**Detect (JSON)**:
```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/detect
```
```json
{
  "detected": true,
  "class": "PESSOA",
  "confidence": 1.0,
  "raw_scores": {"NENHUMA": 0.0, "PESSOA": 1.0}
}
```

**Detect (annotated image)**:
```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/detect/image -o result.jpg
```

**Start camera stream**:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"source": 0}' http://localhost:8000/stream/start
```

**RTSP stream**:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"source": "rtsp://user:pass@camera-ip:554/stream"}' http://localhost:8000/stream/start
```

## Webcam Stream Client

For remote webcam detection over LAN. Run on any machine with a webcam:

```bash
pip install opencv-python requests
python stream_client.py --api http://<server-ip>:8000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--api` | *(required)* | API URL (e.g. `http://<server-ip>:8000`) |
| `--camera` | 0 | Webcam index |
| `--delay` | 1.0 | Seconds between detections |

Opens a window showing the live annotated feed. Press `q` to quit.

## Raspberry Pi Deployment

On the RPi, install only the lightweight inference dependencies (no full TensorFlow):

```bash
pip install tflite-runtime
pip install fastapi uvicorn[standard] python-multipart opencv-python-headless numpy pydantic websockets
```

Copy only these files to the RPi:
- `config.py`
- `api.py`
- `output/person-detect-model.tflite`

```bash
python api.py --model person-detect-model.tflite --port 8000
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Training | TensorFlow/Keras 3, MobileNet |
| Inference | TFLite Runtime (INT8 quantized) |
| API | FastAPI, Uvicorn |
| Image processing | OpenCV |
| Dataset | COCO 2017 |
| Real-time | WebSocket, threading |
