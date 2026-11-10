"""Score outputs/gemini_extra/*_predictions.csv: macro-F1 L1/L2 with bootstrap CIs, by subset.
Usage: python -m arm2_vlm_benchmark.gemini_eval  -> outputs/gemini_extra/RESULTS.md"""
import csv, glob, os
from collections import Counter
import numpy as np
from evaluation.aggregate import macro_f1_boot, encode, ci, L1, L2

OUT = "outputs/gemini_extra"
TRAIN = "data/primary/malayalam/train.csv"


def score(rows, l1c, l2c, n_boot=2000, seed=42):
    n = len(rows)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    res = {"n": n, "unk": sum("UNKNOWN" in (r["pred_level1"], r["pred_level2"]) for r in rows)}
    for k, (key, cl) in enumerate([("level1", l1c), ("level2", l2c)], 1):
        g = encode([r[f"true_{key}"] for r in rows], cl); p = encode([r[f"pred_{key}"] for r in rows], cl)
        full = macro_f1_boot(g, p, len(cl), np.arange(n)[None, :])[0]
        res[f"l{k}"] = (float(full), ci(macro_f1_boot(g, p, len(cl), idx)))
    return res


def maj(rows, key, cl):
    m = Counter(r[f"true_{key}"] for r in rows).most_common(1)[0][0]
    return [dict(r, **{f"pred_{key}": m}) for r in rows], m


def line(name, s):
    return (f"| {name} | {s['n']} | {s['l1'][0]:.3f} [{s['l1'][1][0]:.3f}, {s['l1'][1][1]:.3f}] | "
            f"{s['l2'][0]:.3f} [{s['l2'][1][0]:.3f}, {s['l2'][1][1]:.3f}] | {s['unk']} |")


def main():
    L = ["# Gemini 3.5 Flash-Lite extra runs (Worker A)\n", "Macro-F1, 95% percentile bootstrap (2000 resamples, seed 42). UNKNOWN counted as wrong.\n",
         "| run / subset | n | L1 macro-F1 [CI] | L2 macro-F1 [CI] | UNKNOWN |", "|---|---|---|---|---|"]
    for p in sorted(glob.glob(f"{OUT}/*_predictions.csv")):
        name = os.path.basename(p)[:-16]
        rows = list(csv.DictReader(open(p)))
        subsets = [(name, rows)]
        if name.startswith("zs600"):
            subsets += [(name + " [train-500]", [r for r in rows if r["split"] == "train"]),
                        (name + " [test-100]", [r for r in rows if r["split"] == "test"])]
        for nm, rs in subsets:
            if rs:
                c2 = [c for c in L2 if any(r["true_level2"] == c for r in rs)]  # Tamil has no 'intersection'
                L.append(line(nm + ("" if c2 == L2 else f" (L2 classes present: {'/'.join(c2)})"), score(rs, L1, c2)))
        if name.startswith("tamil"):
            b1, m1 = maj(rows, "level1", L1); b2, m2 = maj(rows, "level2", L2)
            b = [dict(r, pred_level1=m1, pred_level2=m2) for r in rows]
            L.append(line(f"{name} majority baseline ({m1}/{m2}, mode of the Tamil test gold)", score(b, L1, [c for c in L2 if any(r['true_level2']==c for r in rows)])))
    open(f"{OUT}/RESULTS.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
