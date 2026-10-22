"""Web service for Team 14's Malayalam meme triage tool.

Anyone can open the page, upload a meme image and get:
  * level-1 intent (troll vs support) and level-2 target (person / party /
    intersection) from Arm 1 (Swin-Tiny + MuRIL), with class probabilities,
  * Grad-CAM heatmaps for both levels,
  * a "needs human review" flag when the model is not confident.

It is a decision aid for a human moderator, not an automatic classifier: on
the 100-image held-out test set Arm 1 reaches macro-F1 0.51 (level 1) and
0.39 (level 2). Uploaded images are processed in memory and never written
to disk.

Run from the code/ directory:
    .venv/bin/python -m website.server
    open http://127.0.0.1:8000
Env: WEBSITE_DEVICE=cpu|mps|cuda (default cpu), PORT (default 8000),
     HOST (default 127.0.0.1; use 0.0.0.0 to expose on your network).
"""
import base64
import io
import os
import threading

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from arm1_classical.dataset import _EVAL_TRANSFORM, LEVEL1_CLASSES, LEVEL2_CLASSES
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline
from explainability.gradcam_arm1 import compute_gradcam, denormalize, overlay_cam

CHECKPOINT = "outputs/arm1_classical/final_full_train.pt"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000  # decompression-bomb guard
REVIEW_THRESHOLD = 0.60  # below this top-class probability => flag for human review

app = FastAPI(title="Malayalam meme triage")
_lock = threading.Lock()  # one inference at a time; Grad-CAM mutates model grads
_state = {}


def _device():
    return torch.device(os.environ.get("WEBSITE_DEVICE", "cpu"))


def _load():
    if "model" not in _state:
        from transformers import AutoTokenizer
        dev = _device()
        model = ClassicalBaseline().to(dev)
        model.load_state_dict(torch.load(CHECKPOINT, map_location=dev))
        model.eval()
        _state.update(model=model, tok=AutoTokenizer.from_pretrained(TEXT_MODEL_NAME), dev=dev)
    return _state["model"], _state["tok"], _state["dev"]


def _ocr(img):
    try:
        import pytesseract
        return " ".join(pytesseract.image_to_string(img, lang="mal").split()), None
    except Exception as e:  # noqa: BLE001
        return "", f"OCR unavailable ({type(e).__name__}); the text branch received an empty string."


def _png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyse(img, caption):
    model, tok, dev = _load()
    notes = []
    text = (caption or "").strip()
    if text:
        text_src = "your caption"
    else:
        text, err = _ocr(img)
        text_src = "OCR of the image"
        if err:
            notes.append(err)
        elif not text:
            notes.append("No text was found in the image; the text branch received an empty string.")

    img_t = _EVAL_TRANSFORM(img)
    enc = tok(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    ids, mask = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    with torch.no_grad():
        out = model(img_t.unsqueeze(0).to(dev), ids.unsqueeze(0).to(dev), mask.unsqueeze(0).to(dev))

    levels = {}
    for lvl, classes, key in (("level1", LEVEL1_CLASSES, "level1_logits"),
                              ("level2", LEVEL2_CLASSES, "level2_logits")):
        probs = out[key].softmax(1)[0]
        k = int(probs.argmax())
        cam, _, _ = compute_gradcam(model, img_t, ids, mask, dev, level=lvl)
        levels[lvl] = {
            "prediction": classes[k],
            "confidence": float(probs[k]),
            "probabilities": {c: float(p) for c, p in zip(classes, probs.tolist())},
            "needs_review": float(probs[k]) < REVIEW_THRESHOLD,
            "heatmap": _png_b64(overlay_cam(denormalize(img_t), cam)),
        }
    return {
        "levels": levels,
        "needs_review": any(v["needs_review"] for v in levels.values()),
        "text_used": text,
        "text_source": text_src,
        "notes": notes,
    }


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), caption: str = Form("")):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 8 MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        if probe.width * probe.height > MAX_PIXELS:
            raise HTTPException(413, "Image dimensions are too large.")
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not a readable image (use JPG, PNG or WebP).")
    with _lock:
        result = analyse(img, caption[:500])
    return JSONResponse(result)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    _load()  # fail fast on a missing checkpoint instead of on the first request
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
