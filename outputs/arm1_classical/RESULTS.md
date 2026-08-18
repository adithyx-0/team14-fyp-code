# Arm 1 — Classical baseline results

Frozen Swin-Tiny + frozen MuRIL (Malayalam-capable BERT), residual fusion
head, class-weighted CE loss (`data/DECISIONS.md`). Text branch input is
Tesseract OCR text extracted from the meme image (no caption field exists
in the source data — see `data/ocr_extract.py`).

## 5-fold CV on the 500-example train set (`cv_results.json`)
Mean across folds: **level1 macro-F1 = 0.6119, level2 macro-F1 = 0.4550**

## Final eval on the real 100-example held-out test set (`test_results.json`)
Trained once on the full train set, evaluated on `test.csv` — the number
that's actually comparable to SYNAPSE's reported test-set results:

| | Level-1 macro-F1 | Level-2 macro-F1 |
|---|---|---|
| **Arm 1 (this)** | 0.5076 | **0.3940** |
| SYNAPSE paper | 0.9200 | 0.4256 |
| Our SYNAPSE replication (`../test fyp/`) | 0.6564 | 0.3183 |

**Reading this:** Arm 1's classical baseline sits between our own SYNAPSE
replication and the paper's own fine-tuned VLM number on Level-2 — a
legitimate anchor result. Note the CV mean (0.4550) and the single
train/test-split number (0.3940) are *not* the same measurement: CV mean
is averaged over folds carved out of the 500 train examples, the test-set
number is trained on all 500 and evaluated on the true, separate 100-example
held-out set that SYNAPSE's own numbers come from. Report the test-set
number (0.3940) when comparing head-to-head against SYNAPSE; report the CV
mean (0.4550) when characterizing Arm 1's own stability across folds.

Caveats to state in the write-up (same class as the SYNAPSE replication's
own honesty section):
- Only 8 epochs, no LR schedule or hyperparameter search — first working
  run, not a tuned one; likely has headroom.
- OCR text has some noise / ~20% empty extractions (memes with no visible
  text, or OCR misses) — not manually verified against ground truth.
- Backbones (Swin-Tiny, MuRIL) are fully frozen — only the fusion head is
  trained, per the "frozen Swin-Tiny + BERT" spec in instructions.md.
