"""Extract OCR text from meme images for Arm 1's BERT text branch.

The dataset ships with no caption/text field (verified against the raw
xlsx: only meme_id, Level 1, Level 2) — text has to come from the meme
image itself. Uses Tesseract with the Malayalam script pack
(`brew install tesseract-lang`, language code `mal`).

Writes a copy of each split CSV with an added `ocr_text` column, cached so
this slow step (~1 img/sec) only needs to run once.

Usage:
    python data/ocr_extract.py
"""
import csv
import os

import pytesseract
from PIL import Image
from tqdm import tqdm

MALAYALAM_DIR = "data/primary/malayalam"
LANG = "mal"


def ocr_split(split: str):
    in_csv = os.path.join(MALAYALAM_DIR, f"{split}.csv")
    out_csv = os.path.join(MALAYALAM_DIR, f"{split}_ocr.csv")

    with open(in_csv) as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc=f"OCR {split}"):
        img_path = os.path.join(MALAYALAM_DIR, row["image"])
        text = pytesseract.image_to_string(Image.open(img_path), lang=LANG)
        row["ocr_text"] = " ".join(text.split())  # collapse whitespace/newlines

    with open(out_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_empty = sum(1 for r in rows if not r["ocr_text"])
    print(f"{out_csv}: {len(rows)} rows, {n_empty} with empty OCR text")


if __name__ == "__main__":
    ocr_split("train")
    ocr_split("test")
