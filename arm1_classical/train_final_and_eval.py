"""Train Arm 1 once on the full 500-example train set and evaluate on the
real 100-example held-out test set — the fair, apples-to-apples comparison
against SYNAPSE's reported numbers (0.4256 paper / 0.3183 our replication),
both of which are test-set numbers, not CV numbers (instructions.md Phase 7).
"""
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from arm1_classical.dataset import (
    LEVEL1_TO_IDX,
    LEVEL2_TO_IDX,
    MalayalamMemeDataset,
    class_weights,
    load_ocr_csv,
)
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline
from arm1_classical.train import BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, get_device, run_epoch

DATA_ROOT = "data/primary/malayalam"
TRAIN_OCR_CSV = os.path.join(DATA_ROOT, "train_ocr.csv")
TEST_OCR_CSV = os.path.join(DATA_ROOT, "test_ocr.csv")
OUT_DIR = "outputs/arm1_classical"


def main():
    device = get_device()
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    train_rows = load_ocr_csv(TRAIN_OCR_CSV)
    test_rows = load_ocr_csv(TEST_OCR_CSV)

    train_ds = MalayalamMemeDataset(train_rows, DATA_ROOT, tokenizer, train=True)
    test_ds = MalayalamMemeDataset(test_rows, DATA_ROOT, tokenizer, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ClassicalBaseline().to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)

    w_l1 = class_weights(train_rows, "level1", LEVEL1_TO_IDX).to(device)
    w_l2 = class_weights(train_rows, "level2", LEVEL2_TO_IDX).to(device)
    criterion_l1 = nn.CrossEntropyLoss(weight=w_l1)
    criterion_l2 = nn.CrossEntropyLoss(weight=w_l2)

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, criterion_l1, criterion_l2, optimizer)
        print(f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} ({time.time()-t0:.1f}s)")

    test_metrics = run_epoch(model, test_loader, device, criterion_l1, criterion_l2, optimizer=None)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "final_full_train.pt"))
    with open(os.path.join(OUT_DIR, "test_results.json"), "w") as f:
        json.dump(
            {
                "arm1_test_f1_level1": test_metrics["f1_level1"],
                "arm1_test_f1_level2": test_metrics["f1_level2"],
                "synapse_paper_test_f1_level2": 0.4256,
                "synapse_replication_test_f1_level2": 0.3183,
            },
            f,
            indent=2,
        )

    print(f"\ntest set: level1 macro-F1={test_metrics['f1_level1']:.4f} level2 macro-F1={test_metrics['f1_level2']:.4f}")
    print("SYNAPSE paper level2 macro-F1: 0.4256 | our replication: 0.3183")


if __name__ == "__main__":
    main()
