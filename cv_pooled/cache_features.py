"""Cache frozen Swin-Tiny / MuRIL features for all 600 labelled images
(train + test pooled). Image features are cached for the original and the
horizontally flipped view, matching Arm 1's only augmentation.

Usage: python cv_pooled/cache_features.py
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm1_classical.dataset import _EVAL_TRANSFORM, load_ocr_csv  # noqa: E402
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline  # noqa: E402

ROOT = "data/primary/malayalam"
OUT = "outputs/cv_pooled/features.npz"


def main():
    rows = []
    for split in ("train", "test"):
        for r in load_ocr_csv(f"{ROOT}/{split}_ocr.csv"):
            r["split"] = split
            rows.append(r)
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = ClassicalBaseline().eval()  # CPU: deterministic, low RAM
    img, img_flip, txt = [], [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            im = Image.open(os.path.join(ROOT, r["image"])).convert("RGB")
            x = torch.stack([_EVAL_TRANSFORM(im), _EVAL_TRANSFORM(ImageOps.mirror(im))])
            f = model.vision(x)
            img.append(f[0].numpy()); img_flip.append(f[1].numpy())
            enc = tok(r["ocr_text"] or "", padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt")
            t = model.text(**enc).last_hidden_state[:, 0, :]
            txt.append(t[0].numpy())
            if i % 50 == 0:
                print(i, len(rows), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, img=np.stack(img), img_flip=np.stack(img_flip), txt=np.stack(txt),
             image=np.array([r["image"] for r in rows]),
             split=np.array([r["split"] for r in rows]),
             level1=np.array([r["level1"] for r in rows]),
             level2=np.array([r["level2"] for r in rows]))
    print("saved", OUT)


if __name__ == "__main__":
    main()
