"""Dataset for Arm 1 — reads the OCR'd, fold-assigned Malayalam CSVs and
returns (image_tensor, tokenized_ocr_text, level1_label, level2_label).
"""
import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LEVEL1_CLASSES = ["troll", "support"]
LEVEL2_CLASSES = ["person", "party", "intersection"]

LEVEL1_TO_IDX = {c: i for i, c in enumerate(LEVEL1_CLASSES)}
LEVEL2_TO_IDX = {c: i for i, c in enumerate(LEVEL2_CLASSES)}

IMAGE_SIZE = 224  # swin_tiny_patch4_window7_224's native input size

_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MalayalamMemeDataset(Dataset):
    """rows: list[dict] with keys image, level1, level2, ocr_text (paths
    relative to `root`, e.g. as loaded from train_ocr.csv or a fold subset
    of malayalam_train_folds.csv joined with OCR text)."""

    def __init__(self, rows, root, tokenizer, max_text_len=64, train=False):
        self.rows = rows
        self.root = root
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.transform = _TRAIN_TRANSFORM if train else _EVAL_TRANSFORM

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(os.path.join(self.root, row["image"])).convert("RGB")
        img_t = self.transform(img)

        text = row.get("ocr_text", "") or ""
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        level1 = torch.tensor(LEVEL1_TO_IDX[row["level1"]], dtype=torch.long)
        level2 = torch.tensor(LEVEL2_TO_IDX[row["level2"]], dtype=torch.long)

        return {
            "image": img_t,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "level1": level1,
            "level2": level2,
        }


def load_ocr_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def class_weights(rows, key, classes_to_idx):
    """Inverse-frequency class weights for CrossEntropyLoss, per DECISIONS.md
    (class-weighted loss is Arm 1's logged imbalance-handling technique)."""
    counts = torch.zeros(len(classes_to_idx))
    for r in rows:
        counts[classes_to_idx[r[key]]] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(counts) * counts)
    return weights
