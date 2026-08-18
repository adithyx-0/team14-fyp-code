"""Arm 1 — 5-fold CV training of the classical baseline (Swin-Tiny + MuRIL,
residual fusion). This is the anchor number every other arm is compared
against (instructions.md Phase 2) — run this first.

Usage:
    python arm1_classical/train.py
"""
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm1_classical.dataset import (  # noqa: E402
    LEVEL1_CLASSES,
    LEVEL1_TO_IDX,
    LEVEL2_CLASSES,
    LEVEL2_TO_IDX,
    MalayalamMemeDataset,
    class_weights,
    load_ocr_csv,
)
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline  # noqa: E402

FOLDS_CSV = "data/splits/malayalam_train_folds.csv"
OCR_CSV = "data/primary/malayalam/train_ocr.csv"
DATA_ROOT = "data/primary/malayalam"
OUT_DIR = "outputs/arm1_classical"
N_SPLITS = 5

BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-3
NUM_WORKERS = 2


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def merge_folds_and_ocr():
    """malayalam_train_folds.csv has the fold assignment, train_ocr.csv has
    the OCR text — both keyed by image path, join them."""
    fold_rows = load_ocr_csv(FOLDS_CSV)
    ocr_rows = load_ocr_csv(OCR_CSV)
    ocr_by_image = {r["image"]: r["ocr_text"] for r in ocr_rows}

    merged = []
    for r in fold_rows:
        r = dict(r)
        r["ocr_text"] = ocr_by_image[r["image"]]
        merged.append(r)
    return merged


def run_epoch(model, loader, device, criterion_l1, criterion_l2, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_l1_true, all_l1_pred = [], []
    all_l2_true, all_l2_pred = [], []

    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        level1 = batch["level1"].to(device)
        level2 = batch["level2"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(image, input_ids, attention_mask)
            loss_l1 = criterion_l1(out["level1_logits"], level1)
            loss_l2 = criterion_l2(out["level2_logits"], level2)
            loss = loss_l1 + loss_l2

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * image.size(0)
        all_l1_true += level1.cpu().tolist()
        all_l1_pred += out["level1_logits"].argmax(-1).cpu().tolist()
        all_l2_true += level2.cpu().tolist()
        all_l2_pred += out["level2_logits"].argmax(-1).cpu().tolist()

    n = len(loader.dataset)
    f1_l1 = f1_score(all_l1_true, all_l1_pred, average="macro", zero_division=0)
    f1_l2 = f1_score(all_l2_true, all_l2_pred, average="macro", zero_division=0)
    return {"loss": total_loss / n, "f1_level1": f1_l1, "f1_level2": f1_l2}


def run_fold(fold_idx, all_rows, tokenizer, device):
    train_rows = [r for r in all_rows if int(r["fold"]) != fold_idx]
    val_rows = [r for r in all_rows if int(r["fold"]) == fold_idx]

    train_ds = MalayalamMemeDataset(train_rows, DATA_ROOT, tokenizer, train=True)
    val_ds = MalayalamMemeDataset(val_rows, DATA_ROOT, tokenizer, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ClassicalBaseline().to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)

    w_l1 = class_weights(train_rows, "level1", LEVEL1_TO_IDX).to(device)
    w_l2 = class_weights(train_rows, "level2", LEVEL2_TO_IDX).to(device)
    criterion_l1 = nn.CrossEntropyLoss(weight=w_l1)
    criterion_l2 = nn.CrossEntropyLoss(weight=w_l2)

    best_val = None
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, criterion_l1, criterion_l2, optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion_l1, criterion_l2, optimizer=None)
        dt = time.time() - t0
        print(
            f"fold {fold_idx} epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_f1_l1={val_metrics['f1_level1']:.4f} val_f1_l2={val_metrics['f1_level2']:.4f} "
            f"({dt:.1f}s)"
        )
        if best_val is None or (val_metrics["f1_level1"] + val_metrics["f1_level2"]) > (
            best_val["f1_level1"] + best_val["f1_level2"]
        ):
            best_val = val_metrics
            os.makedirs(OUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"fold{fold_idx}_best.pt"))

    return best_val


def main():
    device = get_device()
    print("device:", device)

    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    all_rows = merge_folds_and_ocr()

    results = {}
    for fold_idx in range(N_SPLITS):
        print(f"\n=== fold {fold_idx} ===")
        results[f"fold{fold_idx}"] = run_fold(fold_idx, all_rows, tokenizer, device)

    avg_f1_l1 = sum(r["f1_level1"] for r in results.values()) / N_SPLITS
    avg_f1_l2 = sum(r["f1_level2"] for r in results.values()) / N_SPLITS
    results["mean"] = {"f1_level1": avg_f1_l1, "f1_level2": avg_f1_l2}

    with open(os.path.join(OUT_DIR, "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n5-fold mean: level1 macro-F1={avg_f1_l1:.4f} level2 macro-F1={avg_f1_l2:.4f}")
    print(f"(SYNAPSE paper Malayalam level2 macro-F1: 0.4256; our replication: 0.3183)")


if __name__ == "__main__":
    main()
