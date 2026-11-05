"""Consolidated evaluation across all systems on the frozen 100-example
Malayalam test set (data/primary/malayalam/test.csv).

For every system with per-example predictions on disk it reports macro-F1
(level1, level2, Avg-F1) with 95% bootstrap CIs, paired-bootstrap
differences vs Arm 1, per-class F1, UNKNOWN rate and the level2 confusion
matrix. Systems whose files do not exist yet are listed as "not run".
Paper numbers are included as REPORTED (not reproduced) reference rows.

Usage:  python -m evaluation.aggregate [--n-boot 2000] [--seed 42]
See evaluation/README.md for the rationale behind each choice.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

import numpy as np

DATA_ROOT = "data/primary/malayalam"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
OUT_DIR = "outputs/evaluation"

L1 = ["troll", "support"]
L2 = ["person", "party", "intersection"]
UNK = "UNKNOWN"

# raw label vocabularies seen across systems -> canonical
_NORM = {
    "support": "support", "praise": "support",
    "troll": "troll", "criticise": "troll", "criticize": "troll", "oppose": "troll",
    "person": "person", "party": "party",
    "intersection": "intersection", "both": "intersection",
}

# (display name, path, kind). Fixed systems first; arm2/arm3 *.json are globbed.
FIXED_SYSTEMS = [
    ("Arm 1 — Swin-Tiny + MuRIL (classical)", "outputs/arm1_classical/test_predictions.json"),
    ("SYNAPSE replication — Qwen3-VL-2B LoRA (mlx)", "../test fyp/outputs/predictions_malayalam.json"),
    ("Gemini 3.5 Flash-Lite zero-shot", "outputs/arm2_vlm_benchmark/gemini_zeroshot_predictions.csv"),
]
REFERENCE_ROWS = [
    # reported in the SYNAPSE paper (Malayalam test) — not reproduced here
    {"name": "SYNAPSE paper (reported, not reproduced)", "f1_l1": 0.9200, "f1_l2": 0.4256},
]
ARM1_NAME = FIXED_SYSTEMS[0][0]


def norm(label):
    return _NORM.get(str(label).strip().lower(), UNK)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_gold():
    with open(TEST_CSV) as f:
        rows = list(csv.DictReader(f))
    return [{"id": stem(r["image"]), "l1": r["level1"], "l2": r["level2"]} for r in rows]


def load_predictions(path):
    """-> {image_id: (gold_l1, gold_l2, pred_l1, pred_l2)} with canonical labels, or None if
    the file is not a per-example prediction list."""
    if path.endswith(".csv"):
        with open(path) as f:
            recs = [
                {"image": r["image"], "gold_level1": r["true_level1"], "gold_level2": r["true_level2"],
                 "pred_level1": r["pred_level1"], "pred_level2": r["pred_level2"]}
                for r in csv.DictReader(f)
            ]
    else:
        with open(path) as f:
            recs = json.load(f)
        if not isinstance(recs, list) or not recs or "pred_level1" not in recs[0]:
            return None
    return {
        stem(r["image"]): (norm(r["gold_level1"]), norm(r["gold_level2"]), norm(r["pred_level1"]), norm(r["pred_level2"]))
        for r in recs
    }


def discover_systems():
    systems = list(FIXED_SYSTEMS)
    known = {os.path.abspath(p) for _, p in systems}
    for pat in ("outputs/arm2_vlm_benchmark/*.json", "outputs/arm3_peft_finetune/*.json"):
        for p in sorted(glob.glob(pat)):
            if os.path.abspath(p) in known:
                continue
            systems.append((os.path.splitext(os.path.relpath(p, "outputs"))[0], p))
    return systems


def macro_f1_boot(g, p, k, idx):
    """Vectorised macro-F1 over all bootstrap resamples. g,p: int arrays (p==-1 -> UNKNOWN,
    never matches a class); idx: [B, n] resample indices. Averages over the k gold classes."""
    gb, pb = g[idx], p[idx]
    f1s = []
    for c in range(k):
        tp = ((gb == c) & (pb == c)).sum(1)
        fp = ((gb != c) & (pb == c)).sum(1)
        fn = ((gb == c) & (pb != c)).sum(1)
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1s, axis=0)


def encode(labels, classes):
    m = {c: i for i, c in enumerate(classes)}
    return np.array([m.get(x, -1) for x in labels])


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def per_class_f1(g, p, k):
    out = []
    for c in range(k):
        tp = int(((g == c) & (p == c)).sum()); fp = int(((g != c) & (p == c)).sum()); fn = int(((g == c) & (p != c)).sum())
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return out


def confusion(g, p, k):
    m = np.zeros((k, k + 1), dtype=int)  # last column = UNKNOWN
    for a, b in zip(g, p):
        m[a, b if b >= 0 else k] += 1
    return m


def evaluate_system(name, preds, gold, idx):
    ids = [r["id"] for r in gold]
    missing = [i for i in ids if i not in preds]
    if missing:
        return {"name": name, "status": f"incomplete: {len(missing)}/{len(ids)} test images missing"}
    mismatch = [i for r in gold for i in [r["id"]] if (preds[i][0], preds[i][1]) != (r["l1"], r["l2"])]
    if mismatch:
        return {"name": name, "status": f"gold-label mismatch vs test.csv on {len(mismatch)} images (first: {mismatch[:3]})"}
    g1 = encode([r["l1"] for r in gold], L1); g2 = encode([r["l2"] for r in gold], L2)
    p1 = encode([preds[i][2] for i in ids], L1); p2 = encode([preds[i][3] for i in ids], L2)
    b1 = macro_f1_boot(g1, p1, len(L1), idx); b2 = macro_f1_boot(g2, p2, len(L2), idx)
    full = np.arange(len(ids))[None, :]
    f1_l1 = float(macro_f1_boot(g1, p1, len(L1), full)[0]); f1_l2 = float(macro_f1_boot(g2, p2, len(L2), full)[0])
    return {
        "name": name, "status": "ok", "n": len(ids),
        "f1_l1": f1_l1, "f1_l2": f1_l2, "avg": (f1_l1 + f1_l2) / 2,
        "ci_l1": ci(b1), "ci_l2": ci(b2), "ci_avg": ci((b1 + b2) / 2),
        "boot_l1": b1, "boot_l2": b2,
        "unknown_l1": float((p1 < 0).mean()), "unknown_l2": float((p2 < 0).mean()),
        "per_class_l1": dict(zip(L1, per_class_f1(g1, p1, len(L1)))),
        "per_class_l2": dict(zip(L2, per_class_f1(g2, p2, len(L2)))),
        "confusion_l2": confusion(g2, p2, len(L2)).tolist(),
        "acc_l1": float((g1 == p1).mean()), "acc_l2": float((g2 == p2).mean()),
    }


def majority_baseline(gold):
    with open(TRAIN_CSV) as f:
        rows = list(csv.DictReader(f))
    m1 = Counter(r["level1"] for r in rows).most_common(1)[0][0]
    m2 = Counter(r["level2"] for r in rows).most_common(1)[0][0]
    return {r["id"]: (r["l1"], r["l2"], m1, m2) for r in gold}, m1, m2


def fmt_ci(v, c):
    return f"{v:.4f} [{c[0]:.3f}, {c[1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gold = load_gold()
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(gold), size=(args.n_boot, len(gold)))  # shared across systems => paired

    results, not_run = [], []
    base_preds, m1, m2 = majority_baseline(gold)
    results.append(evaluate_system(f"Majority-class baseline (train mode: {m1}/{m2})", base_preds, gold, idx))
    for name, path in discover_systems():
        if not os.path.exists(path):
            not_run.append((name, path)); continue
        preds = load_predictions(path)
        if preds is None:
            continue  # aggregate-results json, not a per-example file
        results.append(evaluate_system(name, preds, gold, idx))

    ok = [r for r in results if r["status"] == "ok"]
    arm1 = next((r for r in ok if r["name"] == ARM1_NAME), None)

    L = ["# Consolidated evaluation — Malayalam held-out test set (n=%d)\n" % len(gold),
         f"Macro-F1 with 95% percentile-bootstrap CIs ({args.n_boot} resamples of the test set, seed {args.seed}). "
         "UNKNOWN predictions count as wrong. Avg-F1 = mean of level1 and level2 macro-F1. "
         "Seed ± is sample std (ddof=1). Bootstrap convention: a resample with no instance of a class scores that class F1=0 "
         "(affects ~1.7% of L1 resamples, 0.96^100, slightly widening the lower CI).\n",
         "| System | Level-1 macro-F1 [95% CI] | Level-2 macro-F1 [95% CI] | Avg-F1 [95% CI] | UNKNOWN L1/L2 |", "|---|---|---|---|---|"]
    for r in ok:
        L.append(f"| {r['name']} | {fmt_ci(r['f1_l1'], r['ci_l1'])} | {fmt_ci(r['f1_l2'], r['ci_l2'])} | "
                 f"{fmt_ci(r['avg'], r['ci_avg'])} | {r['unknown_l1']:.0%} / {r['unknown_l2']:.0%} |")
    for ref in REFERENCE_ROWS:
        L.append(f"| *{ref['name']}* | {ref['f1_l1']:.4f} (reported) | {ref['f1_l2']:.4f} (reported) | "
                 f"{(ref['f1_l1'] + ref['f1_l2']) / 2:.4f} (reported) | n/a |")
    for r in results:
        if r["status"] != "ok":
            L.append(f"| {r['name']} | — {r['status']} | — | — | — |")
    for name, path in not_run:
        L.append(f"| {name} | not run (no file at `{path}`) | — | — | — |")

    # Seed repeats: report mean +/- std per config instead of letting a single seed stand for it.
    groups = {}
    for r in ok:
        m = re.fullmatch(r"arm3_peft_finetune/(.+)_s(\d+)", r["name"])
        if m:
            groups.setdefault(m.group(1), []).append(r)
    if groups:
        L += ["", "## Arm 3 seed summary (mean ± std over seeds)", "",
              "Single seeds are not reliable at n=100; judge a config by its seed mean, not its best seed. "
              "Seeds were not selected on the test set.", "",
              "| Config | Seeds | Level-1 macro-F1 | Level-2 macro-F1 | Avg-F1 |", "|---|---|---|---|---|"]
        for cfg, rs in sorted(groups.items()):
            cell = lambda k: f"{np.mean([r[k] for r in rs]):.3f} ± {np.std([r[k] for r in rs], ddof=1):.3f}" if len(rs) > 1 else f"{rs[0][k]:.3f} (n=1)"
            L.append(f"| {cfg} | {len(rs)} | {cell('f1_l1')} | {cell('f1_l2')} | {cell('avg')} |")

    sig = []
    if arm1:
        L += ["", "## Paired bootstrap vs Arm 1", "",
              "Same resampled test sets for every system, so each difference is paired. Two-sided p = "
              "2·min(P(Δ≤0), P(Δ≥0)); the CI is on Δ = system − Arm 1. With n=100, differences whose CI "
              "contains 0 are not evidence that either system is better.", "",
              "| System | Δ Level-1 [95% CI] | p | Δ Level-2 [95% CI] | p |", "|---|---|---|---|---|"]
        for r in ok:
            if r is arm1:
                continue
            row = {"name": r["name"]}
            cells = []
            for lv in ("l1", "l2"):
                d = r[f"boot_{lv}"] - arm1[f"boot_{lv}"]
                p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                lo, hi = ci(d)
                row[lv] = {"delta": r[f"f1_{lv}"] - arm1[f"f1_{lv}"], "ci": [lo, hi], "p": float(p)}
                cells.append(f"{r[f'f1_{lv}'] - arm1[f'f1_{lv}']:+.4f} [{lo:+.3f}, {hi:+.3f}] | {p:.3f}")
            L.append(f"| {r['name']} | {cells[0]} | {cells[1]} |")
            sig.append(row)

    L += ["", "## Per-class F1", "", "| System | troll | support | person | party | intersection |", "|---|---|---|---|---|---|"]
    for r in ok:
        pc1, pc2 = r["per_class_l1"], r["per_class_l2"]
        L.append(f"| {r['name']} | {pc1['troll']:.3f} | {pc1['support']:.3f} | {pc2['person']:.3f} | {pc2['party']:.3f} | {pc2['intersection']:.3f} |")
    L += ["", "## Level-2 confusion matrices (rows = gold, columns = predicted)", ""]
    for r in ok:
        L += [f"**{r['name']}**", "", "| gold \\ pred | person | party | intersection | UNKNOWN |", "|---|---|---|---|---|"]
        for c, row in zip(L2, r["confusion_l2"]):
            L.append(f"| {c} | " + " | ".join(map(str, row)) + " |")
        inter_recall = r["confusion_l2"][2][2] / max(1, sum(r["confusion_l2"][2]))
        L += ["", f"intersection recall: {inter_recall:.0%}", ""]
    n_sup = sum(r["l1"] == "support" for r in gold)
    L += ["## Notes", "",
          "- Level-1 is 95%+ troll in train, so accuracy is uninformative; macro-F1 weights the rare class equally (see README).",
          f"- The test set has only {n_sup} `support` examples, so Level-1 macro-F1 rests on {n_sup} images: one image moves it by tens of points, and Level-1 CIs are wide.",
          "- There is no validation split (train 500 / test 100); no config was tuned on a held-out set, and a single best seed picked after seeing test scores is a lead, not a result.",
          "- Level-1 and level-2 test-label distributions come from `test.csv`; every system's own gold columns were checked against it.",
          "- Our macro-F1 averages over the gold classes only, so an UNKNOWN prediction is simply a miss. "
          "(`gemini_zeroshot.py`'s own printed F1 would also average over an 'UNKNOWN' class if any occurred; the numbers coincide when there are none.)",
          "- Paper row is the figure reported in SYNAPSE@DravidianLangTech 2026; it was not re-run here."]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "consolidated.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    slim = [{k: v for k, v in r.items() if not k.startswith("boot_")} for r in results]
    with open(os.path.join(OUT_DIR, "consolidated.json"), "w") as f:
        json.dump({"n_test": len(gold), "n_boot": args.n_boot, "seed": args.seed, "systems": slim,
                   "paired_vs_arm1": sig, "reported_reference": REFERENCE_ROWS,
                   "not_run": [{"name": n, "path": p} for n, p in not_run]}, f, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
