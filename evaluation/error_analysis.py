"""Error analysis: Gemini 3.5 Flash-Lite zero-shot vs Arm 1 on the 100-image
Malayalam test set. Read-only on existing prediction files.
Writes outputs/error_analysis/{error_analysis.md,per_image.csv}.
Example selection is deterministic (first 5 by image id order in test.csv), not cherry-picked.
Usage: python -m evaluation.error_analysis
"""
import csv, json, os
from collections import Counter

G = "outputs/arm2_vlm_benchmark/gemini_zeroshot_predictions.csv"
A = "outputs/arm1_classical/test_predictions.json"
OUT = "outputs/error_analysis"
L2 = ["person", "party", "intersection"]


def main():
    os.makedirs(OUT, exist_ok=True)
    g = {r["image"]: r for r in csv.DictReader(open(G))}
    a = {r["image"]: r for r in json.load(open(A))}
    assert set(g) == set(a) and len(g) == 100
    rows = []
    for img in g:
        r, s = g[img], a[img]
        assert r["true_level2"] == s["gold_level2"] and r["true_level1"] == s["gold_level1"]
        rows.append(dict(image=img, gold_l1=r["true_level1"], gold_l2=r["true_level2"],
                         gem_l1=r["pred_level1"], arm1_l1=s["pred_level1"],
                         gem_l2=r["pred_level2"], arm1_l2=s["pred_level2"],
                         gem_ok_l1=r["pred_level1"] == r["true_level1"], arm1_ok_l1=s["pred_level1"] == s["gold_level1"],
                         gem_ok_l2=r["pred_level2"] == r["true_level2"], arm1_ok_l2=s["pred_level2"] == s["gold_level2"]))
    with open(f"{OUT}/per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

    md = ["# Error analysis: Gemini 3.5 Flash-Lite zero-shot vs Arm 1 (Malayalam test, n=100)", "",
          "Source: `gemini_zeroshot_predictions.csv`, `arm1_classical/test_predictions.json`. Per-image table: `per_image.csv`.", ""]
    ex = {}
    for lv in ("l1", "l2"):
        both = sum(r[f"gem_ok_{lv}"] and r[f"arm1_ok_{lv}"] for r in rows)
        go = sum(r[f"gem_ok_{lv}"] and not r[f"arm1_ok_{lv}"] for r in rows)
        ao = sum(r[f"arm1_ok_{lv}"] and not r[f"gem_ok_{lv}"] for r in rows)
        nb = 100 - both - go - ao
        md += [f"## Level {lv[1]} agreement (correct/incorrect), n=100", "",
               "| | Arm 1 correct | Arm 1 wrong |", "|---|---|---|",
               f"| Gemini correct | {both} | {go} |", f"| Gemini wrong | {ao} | {nb} |", "",
               f"Gemini accuracy {both+go}/100, Arm 1 accuracy {both+ao}/100. Accuracy is shown for the agreement table only; headline metric is macro-F1.", ""]
        ex[lv] = ([r for r in rows if r[f"gem_ok_{lv}"] and not r[f"arm1_ok_{lv}"]],
                  [r for r in rows if r[f"arm1_ok_{lv}"] and not r[f"gem_ok_{lv}"]])
    # L2 per gold class
    md += ["## Level 2 accuracy per gold class", "", "| gold | n | Gemini correct | Arm 1 correct |", "|---|---|---|---|"]
    for c in L2:
        sub = [r for r in rows if r["gold_l2"] == c]
        md.append(f"| {c} | {len(sub)} | {sum(r['gem_ok_l2'] for r in sub)} | {sum(r['arm1_ok_l2'] for r in sub)} |")
    for name, k in (("Gemini", "gem_l2"), ("Arm 1", "arm1_l2")):
        md += ["", f"### {name} level-2 confusion (rows gold, cols predicted)", "", "| gold \\ pred | " + " | ".join(L2) + " |", "|---|---|---|---|"]
        cnt = Counter((r["gold_l2"], r[k]) for r in rows)
        for c in L2:
            md.append(f"| {c} | " + " | ".join(str(cnt[(c, p)]) for p in L2) + " |")
    md += ["", "## Example images (level 2; first 5 by test-file order in each direction; view `data/primary/malayalam/<image>`)", ""]
    for title, lst in (("Gemini right, Arm 1 wrong", ex["l2"][0]), ("Arm 1 right, Gemini wrong", ex["l2"][1])):
        md += [f"### {title} ({len(lst)} total)", "", "| image | gold | Gemini | Arm 1 |", "|---|---|---|---|"]
        md += [f"| {r['image']} | {r['gold_l2']} | {r['gem_l2']} | {r['arm1_l2']} |" for r in lst[:5]]
        md.append("")
    sup = [r for r in rows if r["gold_l1"] == "support"]
    md += ["### Level 1: the 4 support images", "", "| image | Gemini | Arm 1 |", "|---|---|---|"]
    md += [f"| {r['image']} | {r['gem_l1']} | {r['arm1_l1']} |" for r in sup]
    inter = [r for r in rows if r["gold_l2"] == "intersection"]
    gi = Counter(r["gem_l2"] for r in inter); ai = Counter(r["arm1_l2"] for r in inter)
    pi_g = Counter(r["gold_l2"] for r in rows if r["gem_l2"] == "intersection")
    md += ["", "## Why `intersection` is hard", "",
           f"- Only {len(inter)} of 100 test images are `intersection` (train: see class counts in DEVIATIONS/summary), so per-class F1 has very few examples.",
           f"- Gemini predictions on gold-intersection images: {dict(gi)}; Arm 1: {dict(ai)}.",
           f"- When Gemini says `intersection`, the gold labels are: {dict(pi_g)} (precision {pi_g['intersection']}/{sum(pi_g.values())}).",
           "- Observations from the table, not from inspecting every image: `intersection` is defined by a person AND their party being targeted together, which is a judgment about the "
           "relationship between two entities rather than a visual category; `person` vs `intersection` differs by whether party symbols/names are salient and Malayalam text that names the party "
           "is unreadable to a frozen image encoder and (per the pooled CV) the OCR text branch adds nothing. Annotator-boundary ambiguity between the three target classes is plausible but we did not measure inter-annotator agreement.",
           "- Hypotheses above are untested; qualitative review of the listed image ids would confirm or refute them.", ""]
    open(f"{OUT}/error_analysis.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
