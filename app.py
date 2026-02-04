import os
import time
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from inference import load_model, predict_video
from video_preprocess import PreprocessConfig

APP_DIR = Path(__file__).resolve().parent
UI_PATH = APP_DIR / "ui" / "index.html"

app = FastAPI(title="INCLUDE50 Sign Inference")

MODEL = None
LABEL_MAP = None


@app.on_event("startup")
def _load():
    global MODEL, LABEL_MAP
    MODEL, LABEL_MAP = load_model(
        dataset="include50",
        model_type="transformer",
        transformer_size="small",
        checkpoint_path=None,
    )


@app.get("/")
def index():
    html = UI_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None or LABEL_MAP is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    preprocess_config = PreprocessConfig(
        apply_darken=False,
        apply_brighten=True,
        darken_min=0.3,
        darken_max=0.8,
        brighten_method="clahe",
        brighten_gamma_min=1.2,
        brighten_gamma_max=1.8,
    )

    start = time.time()
    try:
        result = predict_video(
            tmp_path,
            dataset="include50",
            model=MODEL,
            label_map=LABEL_MAP,
            preprocess_config=preprocess_config,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    elapsed_ms = int((time.time() - start) * 1000)
    result["elapsed_ms"] = elapsed_ms
    return JSONResponse(result)
