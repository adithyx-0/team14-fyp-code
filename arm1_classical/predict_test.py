"""Save Arm 1's per-example predictions on the 100-example held-out test
set (needed for bootstrap CIs / paired tests in evaluation/aggregate.py;
test_results.json only holds the two aggregate F1 numbers).

Loads outputs/arm1_classical/final_full_train.pt (trained once on the full
500-example train set by train_final_and_eval.py) and does inference only.
Runs on CPU by default so it never competes with GPU/MLX jobs for memory.

Usage:
    python -m arm1_classical.predict_test [--device cpu|mps]
"""
import argparse
import json
import os

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from arm1_classical.dataset import LEVEL1_CLASSES, LEVEL2_CLASSES, MalayalamMemeDataset, load_ocr_csv
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline

DATA_ROOT = "data/primary/malayalam"
TEST_OCR_CSV = os.path.join(DATA_ROOT, "test_ocr.csv")
CHECKPOINT = "outputs/arm1_classical/final_full_train.pt"
OUT_PATH = "outputs/arm1_classical/test_predictions.json"


@torch.no_grad()
def predict(device="cpu"):
    rows = load_ocr_csv(TEST_OCR_CSV)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    ds = MalayalamMemeDataset(rows, DATA_ROOT, tokenizer, train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    model = ClassicalBaseline().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    p1, p2 = [], []
    for b in loader:
        out = model(b["image"].to(device), b["input_ids"].to(device), b["attention_mask"].to(device))
        p1 += out["level1_logits"].argmax(1).tolist()
        p2 += out["level2_logits"].argmax(1).tolist()

    return [
        {
            "image": r["image"],
            "gold_level1": r["level1"],
            "gold_level2": r["level2"],
            "pred_level1": LEVEL1_CLASSES[a],
            "pred_level2": LEVEL2_CLASSES[b],
        }
        for r, a, b in zip(rows, p1, p2)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    recs = predict(args.device)
    with open(OUT_PATH, "w") as f:
        json.dump(recs, f, indent=1)
    f1a = f1_score([r["gold_level1"] for r in recs], [r["pred_level1"] for r in recs], average="macro", zero_division=0)
    f1b = f1_score([r["gold_level2"] for r in recs], [r["pred_level2"] for r in recs], average="macro", zero_division=0)
    print(f"{len(recs)} predictions -> {OUT_PATH}")
    print(f"level1 macro-F1={f1a:.4f} (expected 0.5076)  level2 macro-F1={f1b:.4f} (expected 0.3940)")


if __name__ == "__main__":
    main()
