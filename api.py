"""
FastAPI inference server for person detection using TFLite runtime.

Designed for lightweight deployment (e.g. Raspberry Pi) — uses TFLite runtime
only, not full TensorFlow.

Usage:
  python api.py --model ./output/person-detect-model.tflite
  # or via uvicorn:
  MODEL_PATH=./output/person-detect-model.tflite uvicorn api:app
"""

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from config import CLASSES, CONFIDENCE_THRESHOLD, CAMERA_FPS, IMAGE_DIM

# ---------------------------------------------------------------------------
# TFLite interpreter loading (try lightweight runtime first, fall back to TF)
# ---------------------------------------------------------------------------
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("person-detect")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
interpreter: Optional[Interpreter] = None
input_details = None
output_details = None

# Camera stream state
_stream_thread: Optional[threading.Thread] = None
_stream_stop = threading.Event()
_stream_lock = threading.Lock()
_stream_state = {
    "running": False,
    "source": None,
    "latest": None,  # latest detection result dict
}

# WebSocket clients
_ws_clients: set[WebSocket] = set()
_ws_lock = threading.Lock()

# Event loop reference (set at startup for thread→async bridging)
_loop: Optional[asyncio.AbstractEventLoop] = None

# Config (populated from CLI args or env vars)
_model_path: Optional[str] = None
_confidence_threshold: float = CONFIDENCE_THRESHOLD
_camera_fps: int = CAMERA_FPS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    global interpreter, input_details, output_details
    logger.info(f"Loading TFLite model from {model_path}")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Model loaded. Input: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    logger.info(f"Output: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")


# ---------------------------------------------------------------------------
# Preprocessing & inference
# ---------------------------------------------------------------------------
def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Resize, convert BGR→RGB, and format for the model's expected input type.

    Float32 models: rescale [0,255] → [0,1]
    INT8 models: int8_value = uint8_pixel - 128
    """
    resized = cv2.resize(image_bgr, (IMAGE_DIM, IMAGE_DIM))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.int8:
        # INT8: uint8 [0,255] → int8 [-128,127]
        int8_image = rgb.astype(np.int16) - 128
        return int8_image.astype(np.int8).reshape(1, IMAGE_DIM, IMAGE_DIM, 3)
    else:
        # Float32/Float16: rescale to [0,1] matching training preprocessing
        return (rgb.astype(np.float32) / 255.0).reshape(1, IMAGE_DIM, IMAGE_DIM, 3)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_inference(input_tensor: np.ndarray) -> dict:
    """Run TFLite inference and return detection result."""
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    output_dtype = output_details[0]["dtype"]

    if output_dtype == np.int8:
        # INT8: dequantize using the model's quantization parameters
        out_quant = output_details[0].get("quantization_parameters", {})
        scales = out_quant.get("scales", np.array([1.0]))
        zero_points = out_quant.get("zero_points", np.array([0]))
        dequantized = (output.astype(np.float32) - zero_points[0]) * scales[0]
        probs = _softmax(dequantized)
    else:
        # Float32/Float16: output is already softmax probabilities
        probs = output.astype(np.float32)

    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    detected = predicted_idx == 1 and confidence >= _confidence_threshold
    return {
        "detected": detected,
        "class": CLASSES[predicted_idx],
        "confidence": round(confidence, 4),
        "raw_scores": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
    }


def detect_from_image(image_bgr: np.ndarray) -> dict:
    """Full pipeline: preprocess → inference → result."""
    input_tensor = preprocess_image(image_bgr)
    return run_inference(input_tensor)


# ---------------------------------------------------------------------------
# Camera stream
# ---------------------------------------------------------------------------
def _camera_loop(source):
    """Background thread: capture frames, run inference, broadcast results."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open camera source: {source}")
        with _stream_lock:
            _stream_state["running"] = False
        return

    logger.info(f"Camera stream started (source={source}, fps={_camera_fps})")
    frame_interval = 1.0 / _camera_fps

    while not _stream_stop.is_set():
        start = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame, retrying...")
            time.sleep(0.1)
            continue

        result = detect_from_image(frame)
        result["timestamp"] = time.time()

        with _stream_lock:
            _stream_state["latest"] = result

        # Broadcast to WebSocket clients
        _broadcast_result(result)

        elapsed = time.monotonic() - start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            _stream_stop.wait(timeout=sleep_time)

    cap.release()
    logger.info("Camera stream stopped.")
    with _stream_lock:
        _stream_state["running"] = False
        _stream_state["source"] = None


def _broadcast_result(result: dict):
    """Send result to all connected WebSocket clients (thread-safe)."""
    with _ws_lock:
        clients = list(_ws_clients)

    if not clients or _loop is None:
        return

    async def _send_all():
        disconnected = []
        for ws in clients:
            try:
                await ws.send_json(result)
            except Exception:
                disconnected.append(ws)
        if disconnected:
            with _ws_lock:
                for ws in disconnected:
                    _ws_clients.discard(ws)

    asyncio.run_coroutine_threadsafe(_send_all(), _loop)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class StreamStartRequest(BaseModel):
    source: int | str = 0  # camera index or RTSP/MJPEG URL


class DetectionResponse(BaseModel):
    detected: bool
    class_name: str
    confidence: float
    raw_scores: dict[str, float]


class StreamStatusResponse(BaseModel):
    running: bool
    source: Optional[int | str] = None
    latest: Optional[dict] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()

    # Load model
    model_path = _model_path or os.environ.get("MODEL_PATH", "./output/person-detect-model.tflite")
    if not os.path.isfile(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.error("Train a model first with: python train.py --dataset ./dataset --output ./output")
    else:
        load_model(model_path)

    yield

    # Shutdown: stop camera stream if running
    _stream_stop.set()
    if _stream_thread and _stream_thread.is_alive():
        _stream_thread.join(timeout=5)


app = FastAPI(title="Person Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": interpreter is not None,
        "classes": CLASSES,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Upload an image file and get person detection result."""
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = detect_from_image(image)
    return result


def annotate_image(image_bgr: np.ndarray, result: dict) -> np.ndarray:
    """Draw detection result overlay on the image."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    detected = result["detected"]
    label = result["class"]
    confidence = result["confidence"]

    # Colors: green for PESSOA, red for NENHUM
    color = (0, 200, 0) if detected else (0, 0, 200)

    # Border frame
    thickness = max(2, min(h, w) // 100)
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, thickness)

    # Label background
    text = f"{label} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(h, w) / 600)
    text_thick = max(1, int(font_scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thick)
    cv2.rectangle(annotated, (0, 0), (tw + 16, th + baseline + 16), color, -1)
    cv2.putText(annotated, text, (8, th + 8), font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)

    return annotated


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Upload an image file and get back the annotated image with detection overlay."""
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = detect_from_image(image)
    annotated = annotate_image(image, result)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def ui():
    """Simple web UI for visual testing."""
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Person Detection</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee;
         display: flex; flex-direction: column; align-items: center; padding: 24px; gap: 20px; }
  h1 { font-size: 1.4em; font-weight: 500; }
  .card { background: #1a1a1a; border-radius: 12px; padding: 24px; width: 100%; max-width: 640px; }
  .drop-zone { border: 2px dashed #444; border-radius: 8px; padding: 40px; text-align: center;
               cursor: pointer; transition: border-color .2s; }
  .drop-zone:hover, .drop-zone.drag { border-color: #4a9eff; }
  .drop-zone input { display: none; }
  .result { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; margin-top: 16px; }
  .result img { max-width: 100%; max-height: 400px; border-radius: 8px; }
  .json { background: #0d0d0d; padding: 12px; border-radius: 8px; font-family: monospace;
          font-size: 0.85em; white-space: pre-wrap; width: 100%; margin-top: 8px; }
  .detected { color: #4caf50; } .not-detected { color: #f44336; }
  .loading { opacity: 0.5; pointer-events: none; }
</style>
</head><body>
<h1>Person Detection API</h1>
<div class="card">
  <div class="drop-zone" id="dropZone">
    <p>Drop an image here or click to upload</p>
    <input type="file" id="fileInput" accept="image/*">
  </div>
  <div class="result" id="result"></div>
  <div class="json" id="jsonOut" style="display:none"></div>
</div>
<script>
const dz = document.getElementById('dropZone');
const fi = document.getElementById('fileInput');
const res = document.getElementById('result');
const jo = document.getElementById('jsonOut');

dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); handleFile(e.dataTransfer.files[0]); });
fi.addEventListener('change', () => { if (fi.files[0]) handleFile(fi.files[0]); });

async function handleFile(file) {
  dz.classList.add('loading');
  const fd = new FormData();
  fd.append('file', file);

  const [jsonRes, imgRes] = await Promise.all([
    fetch('/detect', { method: 'POST', body: fd.slice ? new FormData() : fd }),
    fetch('/detect/image', { method: 'POST', body: fd })
  ]);

  // Re-send for JSON (FormData can only be consumed once)
  const fd2 = new FormData();
  fd2.append('file', file);
  const jsonData = await (await fetch('/detect', { method: 'POST', body: fd2 })).json();

  const imgBlob = await imgRes.blob();
  const imgUrl = URL.createObjectURL(imgBlob);

  const cls = jsonData.detected ? 'detected' : 'not-detected';
  res.innerHTML = '<img src="' + imgUrl + '" alt="result">';
  jo.style.display = 'block';
  jo.className = 'json ' + cls;
  jo.textContent = JSON.stringify(jsonData, null, 2);
  dz.classList.remove('loading');
}
</script>
</body></html>"""


@app.post("/stream/start")
def stream_start(req: StreamStartRequest):
    """Start camera/video stream processing."""
    global _stream_thread

    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with _stream_lock:
        if _stream_state["running"]:
            raise HTTPException(status_code=409, detail="Stream already running")

    _stream_stop.clear()

    with _stream_lock:
        _stream_state["running"] = True
        _stream_state["source"] = req.source
        _stream_state["latest"] = None

    _stream_thread = threading.Thread(target=_camera_loop, args=(req.source,), daemon=True)
    _stream_thread.start()

    return {"status": "started", "source": req.source}


@app.post("/stream/stop")
def stream_stop():
    """Stop camera/video stream processing."""
    with _stream_lock:
        if not _stream_state["running"]:
            raise HTTPException(status_code=409, detail="No stream running")

    _stream_stop.set()
    if _stream_thread and _stream_thread.is_alive():
        _stream_thread.join(timeout=5)

    return {"status": "stopped"}


@app.get("/stream/status")
def stream_status():
    with _stream_lock:
        return StreamStatusResponse(
            running=_stream_state["running"],
            source=_stream_state["source"],
            latest=_stream_state["latest"],
        )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time detection results via WebSocket."""
    await websocket.accept()
    with _ws_lock:
        _ws_clients.add(websocket)
    logger.info(f"WebSocket client connected ({len(_ws_clients)} total)")

    try:
        while True:
            # Keep connection alive; client can send pings or messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            _ws_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected ({len(_ws_clients)} total)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Person Detection API Server")
    parser.add_argument("--model", default=None, help="Path to .tflite model file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--fps", type=int, default=CAMERA_FPS)
    args = parser.parse_args()

    _model_path = args.model
    _confidence_threshold = args.confidence
    _camera_fps = args.fps

    uvicorn.run(app, host=args.host, port=args.port)
