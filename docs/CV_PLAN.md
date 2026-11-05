# Cross-validation plan (not run; planning only)

Goal: shrink the uncertainty that a 100-image test set (only 4 support
images) leaves on every comparison. Arm 1 already does 5-fold CV on the 500
train images (`arm1_classical/train.py`, mean L1 0.612 / L2 0.455, folds from
`data/splits/malayalam_train_folds.csv`, stratified on level 1). The plan
below extends that to all 600 labelled images with repeats.

## What the Arm 1 script does today (read from the code)
- Both backbones are frozen and kept in eval mode (`ClassicalBaseline.train()`
  re-asserts this). Only `vision_proj`, `text_proj`, 2 fusion blocks and the
  two heads train (about 1.6M parameters).
- 8 epochs, batch 16, AdamW lr 1e-3, class-weighted cross-entropy, 5 folds.
- The only augmentation is a random horizontal flip (`_TRAIN_TRANSFORM`).
- Observed cost: about 5.5 minutes per fold on MPS (checkpoint timestamps
  23:54 to 00:16 for folds 0 to 4). Almost all of that is re-running the
  frozen Swin and MuRIL every epoch.

## Can the frozen features be cached? Yes, exactly
- **Text:** MuRIL is frozen, in eval mode, and the input is fixed OCR text, so
  its [CLS] vector is deterministic. Cache once per image (768 floats).
- **Image:** Swin pooled output is deterministic given the pixels. The only
  random augmentation is a horizontal flip, so there are exactly two possible
  inputs per image. Cache both (original and flipped, 2 x 768 floats) and have
  each epoch pick one at random per image. That reproduces the current
  training distribution, not an approximation of it.
- Training the head then needs no backbone passes: each epoch is a few
  milliseconds on CPU.
- To keep this safe, first run one fold both ways and check the cached
  version gives the same predictions up to float tolerance.

## Cost estimate (estimates, not measurements)
| Item | Estimate |
|---|---|
| Cache size, 600 images | 600 x (2 x 768 + 768) floats, about 5 MB |
| One-time feature extraction | Swin on 1,200 image views plus MuRIL on 600 texts: a few minutes on CPU, less on MPS |
| Peak RAM during extraction | about 1.5 to 2 GB (MuRIL base is about 0.95 GB in fp32, plus Swin and activations) |
| One 5-fold run on cached features | seconds |
| 10 repeats x 5 folds | well under 10 minutes |
| Peak RAM during head training | under 0.5 GB |

The extraction step exceeds a 1 GB budget, so run it only when the LoRA jobs
have finished or the Mac is otherwise idle.

## Design
1. Pool train and test (600 images). Repeated stratified 5-fold, stratified on
   level 1 (27 support images in total, so 5 to 6 per fold). Joint
   stratification is not possible: `support|intersection` has one example.
2. 10 repeats with different seeds. Pool the out-of-fold predictions per
   repeat and compute macro-F1 on all 600, so the support class has 27 cases
   instead of 4.
3. Report mean and spread across repeats, plus a bootstrap CI over images.
4. Keep the official 100-image test result as a separate row. Pooling means
   CV numbers are not held-out test numbers; do not mix them in one table
   column, and do not tune hyperparameters on the CV folds and then report
   the same folds as a final number.

## Extending it to the other arms
- **Arm 2 zero-shot / CoT:** no training, so it can be scored on all 600
  images. Cost is inference only (100 images already ran; scale by 6).
  Needs the Mac idle (MLX).
- **Arm 3 LoRA:** needs a retrain per fold. At about 10 minutes per run that is
  about 50 minutes for one 5-fold repeat, plus inference per fold, so at most
  one repeat is realistic on this hardware. Do Arm 3 last.
- **Gemini:** inference only, API cost and rate limits apply.

## Caveat on comparability
Cross-validated numbers measure each system on a broader sample, which is
good, but they are not the numbers the SYNAPSE paper reports. Whether the
paper's split matches our 100-image test set is still unverified
(`DEVIATIONS.md`, "Level-1 comparability").

## Order of work
1. Write a cache-building script and a head-only trainer (new files; do not
   modify `train.py` in place).
2. Verify equivalence on one fold.
3. Run 10 x 5-fold on 600 images and add a row to `evaluation/` outputs.
4. Zero-shot on all 600, then decide about Arm 3.
