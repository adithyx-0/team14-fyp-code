"""Grad-CAM for Arm 1 (frozen Swin-Tiny + MuRIL, residual fusion).

Swin-Tiny has no single "last conv layer" the way a CNN does, but timm's
`forward_features()` returns the last stage's output before pooling —
shape [B, 7, 7, 768] for 224x224 input (each 7x7 cell = a 32x32-px patch of
the input image). That is treated as the Grad-CAM target layer, same as a
CNN's last conv feature map: capture it + its gradient w.r.t. the predicted
class logit, weight channels by their gradient's spatial average, ReLU,
upsample to 224x224, overlay on the original image.

The vision backbone is frozen (requires_grad=False on its parameters), so
by default no gradient would reach the feature map — the classifier layer
in the loss chain only needs input grad, not weight grad. Fix: mark the
input image tensor itself as requires_grad=True; that alone is enough for
autograd to build a graph from image -> frozen-conv-ops -> feats, even
though the conv weights don't accumulate gradients.

Usage:
    python -m explainability.gradcam_arm1 [--n 12] [--level level2]
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

from arm1_classical.dataset import (
    LEVEL1_CLASSES,
    LEVEL2_CLASSES,
    _EVAL_TRANSFORM,
)
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline

DATA_ROOT = "data/primary/malayalam"
TEST_OCR_CSV = os.path.join(DATA_ROOT, "test_ocr.csv")
CHECKPOINT = "outputs/arm1_classical/final_full_train.pt"
OUT_DIR = "outputs/explainability/gradcam"

# ImageNet normalization used by _EVAL_TRANSFORM — needed to undo it for display
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def jet_colormap(x):
    """x: float array in [0,1], any shape. Returns uint8 RGB array (*x.shape, 3).
    Hand-rolled jet approximation (no matplotlib/opencv in this venv)."""
    r = np.clip(np.minimum(4 * x - 1.5, -4 * x + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * x - 0.5, -4 * x + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * x + 0.5, -4 * x + 2.5), 0, 1)
    return np.stack([r, g, b], axis=-1)


def denormalize(img_t):
    """img_t: [3, H, W] normalized tensor -> uint8 HWC numpy array."""
    arr = img_t.permute(1, 2, 0).cpu().numpy()
    arr = arr * _STD + _MEAN
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def compute_gradcam(model, img_t, input_ids, attention_mask, device, level="level2"):
    """Returns (cam [7,7] float in [0,1], pred_idx, class_names)."""
    class_names = LEVEL1_CLASSES if level == "level1" else LEVEL2_CLASSES
    head = model.level1_head if level == "level1" else model.level2_head

    img_t = img_t.unsqueeze(0).to(device).requires_grad_(True)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    feats = model.vision.forward_features(img_t)  # [1, 7, 7, 768], grad-enabled via input
    feats.retain_grad()
    pooled_img = feats.mean(dim=(1, 2))  # matches timm's avg global pool -> [1, 768]

    with torch.no_grad():
        text_out = model.text(input_ids=input_ids, attention_mask=attention_mask)
    text_feat = text_out.last_hidden_state[:, 0, :]

    img_proj = model.vision_proj(pooled_img)
    text_proj = model.text_proj(text_feat)
    fused = img_proj + text_proj
    for block in model.fusion_blocks:
        fused = block(fused)

    logits = head(fused)
    pred_idx = logits.argmax(dim=1).item()

    model.zero_grad(set_to_none=True)
    logits[0, pred_idx].backward()

    grads = feats.grad[0]  # [7, 7, 768]
    activ = feats.detach()[0]  # [7, 7, 768]
    weights = grads.mean(dim=(0, 1))  # [768] — GAP of gradient per channel

    cam = torch.einsum("hwc,c->hw", activ, weights)
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), pred_idx, class_names


def overlay_cam(img_uint8, cam_7x7, alpha=0.45):
    """img_uint8: [224,224,3] uint8. cam_7x7: [7,7] float in [0,1]."""
    cam_img = Image.fromarray((cam_7x7 * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0
    heat = (jet_colormap(cam_resized) * 255).astype(np.uint8)
    blended = (img_uint8.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="number of test images to run (spread across classes)")
    ap.add_argument("--level", choices=["level1", "level2"], default="level1")
    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    with open(TEST_OCR_CSV) as f:
        rows = list(csv.DictReader(f))

    # spread the sample across level1 classes so the guide sees both, not
    # just whichever class happens to sort first in the CSV
    by_class = {}
    for r in rows:
        by_class.setdefault(r["level1"], []).append(r)
    picked = []
    i = 0
    classes = list(by_class.keys())
    while len(picked) < min(args.n, len(rows)):
        cls = classes[i % len(classes)]
        if by_class[cls]:
            picked.append(by_class[cls].pop(0))
        i += 1
        if all(not v for v in by_class.values()):
            break

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = ClassicalBaseline().to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)

    for row in picked:
        img_path = os.path.join(DATA_ROOT, row["image"])
        img = Image.open(img_path).convert("RGB")
        img_t = _EVAL_TRANSFORM(img)

        text = row.get("ocr_text", "") or ""
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        cam, pred_idx, class_names = compute_gradcam(
            model, img_t, input_ids, attention_mask, device, level=args.level
        )
        pred_label = class_names[pred_idx]
        true_label = row[args.level]

        disp_img = denormalize(img_t)
        overlay = overlay_cam(disp_img, cam)

        stem = os.path.splitext(os.path.basename(row["image"]))[0]
        correct = "correct" if pred_label == true_label else "wrong"
        out_path = os.path.join(OUT_DIR, f"{stem}_{args.level}_true-{true_label}_pred-{pred_label}_{correct}.png")
        overlay.save(out_path)
        print(f"{row['image']}: true={true_label} pred={pred_label} ({correct}) -> {out_path}")

    print(f"\n{len(picked)} Grad-CAM overlays written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
