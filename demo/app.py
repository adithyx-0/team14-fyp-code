"""Gradio demo for Team 14's meme-stance pipeline.

Design rules (instructions.md Phase 8 / Risk 5):
  * Model choice is a VISIBLE dropdown. There is no automatic "best model":
    with several systems x 2 label levels, "best" is ambiguous and picking
    one silently would pre-empt the comparison this project makes.
  * Arm 1 (Swin-Tiny + MuRIL) runs LIVE on any uploaded meme, with Grad-CAM
    overlays for both label levels.
  * Other systems are not run live on this machine. If the uploaded image is
    one of the 100 held-out test memes (matched by file hash) their
    precomputed test-set prediction is shown; otherwise the demo says so
    plainly. Nothing is ever fabricated.

Usage:
    python -m demo.app            # launch on http://127.0.0.1:7860
    python -m demo.app --smoke    # headless self-test, exits 0 on success
Env: DEMO_DEVICE=cpu|mps|cuda (default cpu, so it never competes with MLX jobs).
"""
import argparse
import csv
import hashlib
import os
import sys

import gradio as gr
import torch
from PIL import Image

from arm1_classical.dataset import _EVAL_TRANSFORM, LEVEL1_CLASSES, LEVEL2_CLASSES
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline
from evaluation.aggregate import FIXED_SYSTEMS, discover_systems, load_predictions, norm

try:  # Grad-CAM lives in explainability/ — degrade gracefully if it fails to import, never crash
    from explainability.gradcam_arm1 import compute_gradcam, denormalize, overlay_cam
except Exception as _e:  # noqa: BLE001
    compute_gradcam = None
    _GRADCAM_ERR = repr(_e)

DATA_ROOT = "data/primary/malayalam"
TEST_OCR_CSV = os.path.join(DATA_ROOT, "test_ocr.csv")
CHECKPOINT = "outputs/arm1_classical/final_full_train.pt"
ARM1_LABEL = f"{FIXED_SYSTEMS[0][0]} — live inference"
PRE_SUFFIX = " — precomputed (test set only)"
LEVEL_NAMES = {"troll": "troll / oppose", "support": "support / praise"}

_state = {}


def _device():
    return torch.device(os.environ.get("DEMO_DEVICE", "cpu"))


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _test_index():
    """{md5 of test image bytes: row} so an uploaded known test meme is recognised."""
    if "test_index" not in _state:
        with open(TEST_OCR_CSV) as f:
            rows = list(csv.DictReader(f))
        _state["test_index"] = {_md5(os.path.join(DATA_ROOT, r["image"])): r for r in rows}
    return _state["test_index"]


def _precomputed():
    if "pre" not in _state:
        pre = {}
        for name, path in discover_systems():
            if name == FIXED_SYSTEMS[0][0] or not os.path.exists(path):
                continue
            preds = load_predictions(path)
            if preds:
                pre[name + PRE_SUFFIX] = preds
        _state["pre"] = pre
    return _state["pre"]


def choices():
    return [ARM1_LABEL] + list(_precomputed().keys())


def _arm1():
    if "arm1" not in _state:
        from transformers import AutoTokenizer
        dev = _device()
        model = ClassicalBaseline().to(dev)
        model.load_state_dict(torch.load(CHECKPOINT, map_location=dev))
        model.eval()
        _state["arm1"] = (model, AutoTokenizer.from_pretrained(TEXT_MODEL_NAME), dev)
    return _state["arm1"]


def _ocr(path):
    try:
        import pytesseract
        return " ".join(pytesseract.image_to_string(Image.open(path), lang="mal").split()), None
    except Exception as e:  # noqa: BLE001
        return "", f"OCR unavailable ({type(e).__name__}); text branch received an empty string."


def run_arm1(path, caption):
    model, tok, dev = _arm1()
    notes = []
    row = _test_index().get(_md5(path))
    text = (caption or "").strip()
    if text:
        src = "your caption"
    elif row is not None:
        text, src = row["ocr_text"], "stored OCR text of this test meme"
    else:
        text, err = _ocr(path)
        src = "Tesseract OCR of the image"
        if err:
            notes.append(err)
    img_t = _EVAL_TRANSFORM(Image.open(path).convert("RGB"))
    enc = tok(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    ids, mask = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
    with torch.no_grad():
        out = model(img_t.unsqueeze(0).to(dev), ids.unsqueeze(0).to(dev), mask.unsqueeze(0).to(dev))
    lines, overlays = [], []
    for lvl, classes, key in (("level1", LEVEL1_CLASSES, "level1_logits"), ("level2", LEVEL2_CLASSES, "level2_logits")):
        probs = out[key].softmax(1)[0]
        k = int(probs.argmax())
        lines.append(f"**{lvl}:** {classes[k]} ({probs[k]:.0%} confidence; "
                     + ", ".join(f"{c} {p:.0%}" for c, p in zip(classes, probs.tolist())) + ")")
        if compute_gradcam is None:
            overlays.append(None)
            notes.append(f"Grad-CAM unavailable: {_GRADCAM_ERR}")
        else:
            cam, _, _ = compute_gradcam(model, img_t, ids, mask, dev, level=lvl)
            overlays.append(overlay_cam(denormalize(img_t), cam))
    md = ["### Arm 1 — live prediction", *lines, "", f"Text branch input: {src}" + (f" — “{text[:80]}”" if text else " (empty)")]
    if row is not None:
        md.append(f"This is held-out test meme `{row['image']}` — gold: level1 **{row['level1']}**, level2 **{row['level2']}**.")
    md += [f"> {n}" for n in dict.fromkeys(notes)]
    return "\n\n".join(md), overlays[0], overlays[1]


def run_precomputed(path, system):
    row = _test_index().get(_md5(path))
    if row is None:
        return ("### Live inference not available on this machine for this system\n\n"
                f"**{system.replace(PRE_SUFFIX, '')}** was evaluated offline on the 100 held-out test memes only. "
                "The uploaded image is not one of them, so there is no result to show. "
                "Choose the Arm 1 entry for live inference, or upload a test-set meme."), None, None
    sid = os.path.splitext(os.path.basename(row["image"]))[0]
    g1, g2, p1, p2 = _precomputed()[system][sid]
    md = (f"### {system.replace(PRE_SUFFIX, '')} — precomputed test-set result\n\n"
          f"Live inference is not available here; showing the stored prediction for test meme `{row['image']}`.\n\n"
          f"**level1:** {p1} (gold {g1})  \n**level2:** {p2} (gold {g2})\n\n"
          "No confidence scores or Grad-CAM are stored for this system.")
    return md, None, None


def predict(path, caption, system):
    if not path:
        return "Upload a meme image first.", None, None
    if system == ARM1_LABEL:
        return run_arm1(path, caption)
    return run_precomputed(path, system)


def build():
    with gr.Blocks(title="Team 14 — Malayalam meme stance") as demo:
        gr.Markdown("# Malayalam political meme analysis — Team 14\n"
                    "Choose the system yourself; no model is auto-selected. Level-1: troll vs support. "
                    "Level-2: person / party / intersection.")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath", label="Meme image")
                cap = gr.Textbox(label="Caption / meme text (optional — Arm 1 falls back to OCR)")
                sysd = gr.Dropdown(choices(), value=ARM1_LABEL, label="System")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                res = gr.Markdown()
                with gr.Row():
                    o1 = gr.Image(label="Grad-CAM — level1", interactive=False)
                    o2 = gr.Image(label="Grad-CAM — level2", interactive=False)
        btn.click(predict, [img, cap, sysd], [res, o1, o2])
    return demo


def smoke():
    with open(TEST_OCR_CSV) as f:
        row = next(csv.DictReader(f))
    path = os.path.join(DATA_ROOT, row["image"])
    md, o1, o2 = predict(path, "", ARM1_LABEL)
    assert "level1" in md and "level2" in md, md
    if compute_gradcam is not None:
        assert o1 is not None and o2 is not None, "Grad-CAM overlays missing"
    print(md)
    for s in choices()[1:]:
        m, _, _ = predict(path, "", s)
        assert "precomputed test-set result" in m, m
        print("OK:", s)
    novel = os.path.join(os.environ.get("DEMO_SMOKE_TMP", "/tmp"), "demo_smoke_novel.png")
    Image.open(path).convert("RGB").resize((300, 300)).save(novel)  # not a test-set file => unknown-image path
    md2, a1, a2 = predict(novel, "", ARM1_LABEL)
    assert "live prediction" in md2 and "held-out test meme" not in md2
    if choices()[1:]:
        assert "not available on this machine" in predict(novel, "", choices()[1])[0]
    os.remove(novel)
    assert predict(None, "", ARM1_LABEL)[0].startswith("Upload")
    demo = build()
    demo.launch(prevent_thread_lock=True, quiet=True, server_name="127.0.0.1")
    demo.close()
    print("SMOKE OK (choices: %d)" % len(choices()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        smoke()
        sys.exit(0)
    build().launch(server_name="127.0.0.1")
