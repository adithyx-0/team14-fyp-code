# Data layer decisions (Phase 1)

Frozen here, once, shared by all three arms — per instructions.md Phase 1
and Risk 3 (imbalance handling must not diverge silently between arms).

## 5-fold split
- `data/splits/malayalam_train_folds.csv` — the 500-row primary train set
  with a `fold` column (0-4), stratified on `level1` (support/troll).
- Stratified on `level1` only, not the joint `(level1, level2)` label:
  `support|intersection` has just 1 example train-wide, which breaks
  5-fold stratification on the joint label. `level1` is the axis with the
  worse imbalance (23 support / 477 troll) and is what actually needs a
  guaranteed-even split; Level-2 spread across folds is a by-product,
  logged in the script's output, not guaranteed.
- Seed 42. Regenerate with `python data/make_splits.py` — do not hand-edit
  the output; if the split needs to change, change the script and rerun.
- `data/primary/malayalam/test.csv` (100 rows) is the frozen held-out test
  set, untouched by CV — matches the paper's own train/test split.

## Augmentation conditions (none / manual / LLM-synthetic)
**Status: deferred to Phase II.** With Mid Review on 21 Aug 2026 and this
session starting 15 Aug (6-day runway), building and validating a
three-condition augmentation matrix before Arm 1 exists isn't worth the
schedule risk. Per instructions.md Section 5's MVP scope-down, Arm 1 uses
the **none** condition (raw 5-fold split above) for Mid Review.

Imbalance handling for Mid Review is instead pushed into each arm's own
training loss, logged here per arm rather than decided independently:
- **Arm 1:** class-weighted loss (weights inverse to `level1`/`level2`
  frequency in the training fold) — see arm1_classical/.
- **Arm 2 / Arm 3:** not yet built; log their imbalance-handling choice
  here when built (few-shot exemplar selection for Arm 2, per the design
  review's original risk framing).

Revisit manual and LLM-synthetic augmentation post–Mid Review, per
instructions.md Phase 1 and the design review's Risk 3.
