"""Manager add-on (10 Nov 2026): tables for the systems added on 10 Nov, WITHOUT touching
outputs/evaluation/consolidated.{md,json}. Reuses the metric/bootstrap code from evaluation.aggregate.
Usage: python -m evaluation.new_results_10nov  -> outputs/evaluation/NEW_RESULTS_10NOV.md
"""
import csv
import glob
import json
import os

import numpy as np

from evaluation.aggregate import (L1, L2, ARM1_NAME, FIXED_SYSTEMS, encode, ci, load_gold,
                                  load_predictions, macro_f1_boot, norm, per_class_f1, stem)

OUT = "outputs/evaluation/NEW_RESULTS_10NOV.md"
N_BOOT, SEED = 2000, 42


def test100_systems():
    """name -> {id: (g1, g2, p1, p2)} for the 100-image Malayalam test set."""
    s = {"Arm 1 (reference)": load_predictions(FIXED_SYSTEMS[0][1])}
    s["Gemini 3.5 Flash-Lite zero-shot (2nd run, from zs600 test split)"] = _filter_split(
        "outputs/gemini_extra/zs600_predictions.csv", "test")
    s["Gemini 3.5 Flash-Lite CoT"] = _filter_split("outputs/gemini_extra/cot_predictions.csv", "test")
    s["Gemini 3.5 Flash-Lite few-shot (6 train exemplars)"] = _filter_split(
        "outputs/gemini_extra/fewshot_predictions.csv", "test")
    p = "outputs/paligemma/paligemma2_3b_zeroshot.json"
    if os.path.exists(p):
        s["PaliGemma2-3B-mix-448 zero-shot (8-bit MLX)"] = load_predictions(p)
    return {k: v for k, v in s.items() if v}


def _filter_split(path, split):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return {stem(r["image"]): (norm(r["true_level1"]), norm(r["true_level2"]),
                               norm(r["pred_level1"]), norm(r["pred_level2"])) for r in rows}


def score(preds, ids, idx):
    g1 = encode([preds[i][0] for i in ids], L1); g2 = encode([preds[i][1] for i in ids], L2)
    p1 = encode([preds[i][2] for i in ids], L1); p2 = encode([preds[i][3] for i in ids], L2)
    full = np.arange(len(ids))[None, :]
    b1, b2 = macro_f1_boot(g1, p1, 2, idx), macro_f1_boot(g2, p2, 3, idx)
    return dict(f1=(float(macro_f1_boot(g1, p1, 2, full)[0]), float(macro_f1_boot(g2, p2, 3, full)[0])),
                b1=b1, b2=b2, ci1=ci(b1), ci2=ci(b2), cia=ci((b1 + b2) / 2),
                unk=((p1 < 0).mean(), (p2 < 0).mean()),
                pc2=dict(zip(L2, per_class_f1(g2, p2, 3))))


def main():
    gold = load_gold(); ids = [r["id"] for r in gold]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    systems = test100_systems()
    for n, p in systems.items():  # gold must match test.csv
        for r in gold:
            assert (p[r["id"]][0], p[r["id"]][1]) == (r["l1"], r["l2"]), (n, r["id"])
    res = {n: score(p, ids, idx) for n, p in systems.items()}
    a1 = res["Arm 1 (reference)"]
    L = ["# New results, 10 Nov 2026 (Manager-assembled; consolidated.md/json untouched)\n",
         "Same 100-image Malayalam test set, same bootstrap (2000 resamples, seed 42, percentile 95% CI) as "
         "`consolidated.md`. UNKNOWN counted as wrong. Numbers recomputed from raw prediction files.\n",
         "## A. Malayalam test (n=100)\n",
         "| System | L1 macro-F1 [CI] | L2 macro-F1 [CI] | Avg | UNK L1/L2 | dAvg vs Arm 1 [CI] | L2 person/party/intersection F1 |",
         "|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        d = ""
        if n != "Arm 1 (reference)":
            diff = (r["b1"] + r["b2"]) / 2 - (a1["b1"] + a1["b2"]) / 2
            lo, hi = ci(diff)
            d = f"{(sum(r['f1']) - sum(a1['f1'])) / 2:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        L.append(f"| {n} | {r['f1'][0]:.3f} [{r['ci1'][0]:.3f}, {r['ci1'][1]:.3f}] | "
                 f"{r['f1'][1]:.3f} [{r['ci2'][0]:.3f}, {r['ci2'][1]:.3f}] | {sum(r['f1'])/2:.3f} "
                 f"[{r['cia'][0]:.3f}, {r['cia'][1]:.3f}] | {r['unk'][0]:.0%} / {r['unk'][1]:.0%} | {d} | "
                 f"{r['pc2']['person']:.2f} / {r['pc2']['party']:.2f} / {r['pc2']['intersection']:.2f} |")
    L.append("\nPaired differences use the same resample indices for both systems (paired bootstrap). "
             "The test set has only 4 `support` items (96 troll), so L1 CIs are wide.\n")
    # extra sections appended from the workers' own result files (copied verbatim, labelled)
    for title, path in [("B. Gemini extra runs: worker's RESULTS.md (600-image zero-shot, Tamil; verbatim)",
                         "outputs/gemini_extra/RESULTS.md"),
                        ("C. Tamil transfer of Arm 1 (Worker B, verbatim JSON)",
                         "outputs/cross_lingual/tamil_transfer_results.json")]:
        if os.path.exists(path):
            txt = open(path).read()
            L += [f"## {title}\n", txt if path.endswith(".md") else "```json\n" + txt + "\n```", ""]
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
