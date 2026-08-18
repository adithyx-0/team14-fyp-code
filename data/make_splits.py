"""Build the canonical, frozen 5-fold stratified split for the primary
Malayalam dataset. Run once; all three arms read the resulting folds.csv
rather than splitting independently (instructions.md Phase 1 / Risk 3).

Stratifies on level1 (the severely imbalanced stance label: 23 support /
477 troll in the train split) since StratifiedKFold needs >=5 examples per
class for 5 folds, and the joint (level1, level2) label has a singleton
class (support|intersection: 1 example) that breaks joint stratification.
Level-2 distribution across folds is reported for visibility but not
guaranteed even.

Usage:
    python data/make_splits.py
"""
import csv

from sklearn.model_selection import StratifiedKFold

TRAIN_CSV = "data/primary/malayalam/train.csv"
OUT_CSV = "data/splits/malayalam_train_folds.csv"
N_SPLITS = 5
SEED = 42


def main():
    with open(TRAIN_CSV) as f:
        rows = list(csv.DictReader(f))

    strat_labels = [r["level1"] for r in rows]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = [-1] * len(rows)
    for fold_idx, (_, val_idx) in enumerate(skf.split(rows, strat_labels)):
        for i in val_idx:
            fold_of[i] = fold_idx

    assert all(f != -1 for f in fold_of)

    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) + ["fold"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r, fold in zip(rows, fold_of):
            r["fold"] = fold
            w.writerow(r)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")

    # sanity: per-fold joint-label distribution
    from collections import Counter

    per_fold = [Counter() for _ in range(N_SPLITS)]
    for r, fold in zip(rows, fold_of):
        per_fold[fold][f"{r['level1']}|{r['level2']}"] += 1
    for i, c in enumerate(per_fold):
        print(f"fold {i}: {dict(c)}")


if __name__ == "__main__":
    main()
