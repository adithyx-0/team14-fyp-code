"""Cross-lingual checkpoint transfer: Malayalam-trained Arm 1 (read-only
final_full_train.pt) evaluated zero-shot on Tamil POLITICAL memes (not the
misogyny set). OCR with Tesseract `tam` (+eng). Resumable OCR cache.
Usage: python -m cross_lingual.tamil_transfer
"""
import csv, json, os
import numpy as np, torch, pytesseract
from PIL import Image
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from arm1_classical.dataset import LEVEL1_CLASSES, LEVEL2_CLASSES, _EVAL_TRANSFORM
from arm1_classical.model import TEXT_MODEL_NAME, ClassicalBaseline

TEST = "/Users/adithya/Final Year Project/test fyp/data/tamil/test.csv"
CKPT = "outputs/arm1_classical/final_full_train.pt"
OUT = "outputs/cross_lingual"
MAP1 = {"troll": "troll", "support": "support"}
MAP2 = {"person": "person", "party": "party", "intersection": "intersection"}


def boot_f1(g, p, labels, n=2000, seed=42):
    rng = np.random.default_rng(seed); g = np.array(g); p = np.array(p); N = len(g); v = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        v.append(f1_score(g[i], p[i], labels=labels, average="macro", zero_division=0))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(TEST)))
    for r in rows:
        if r["level1"] not in MAP1: raise ValueError(f"unknown L1 {r['level1']}")
        if r["level2"] not in MAP2: raise ValueError(f"unknown L2 {r['level2']}")
        if not os.path.exists(r["image"]): raise FileNotFoundError(r["image"])
    cache_p = f"{OUT}/tamil_ocr_cache.json"
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    for i, r in enumerate(rows):
        if r["image"] not in cache:
            cache[r["image"]] = pytesseract.image_to_string(Image.open(r["image"]).convert("RGB"), lang="tam+eng").strip()
            if i % 20 == 0:
                json.dump(cache, open(cache_p, "w"), ensure_ascii=False); print("ocr", i, flush=True)
    json.dump(cache, open(cache_p, "w"), ensure_ascii=False)
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = ClassicalBaseline(); model.load_state_dict(torch.load(CKPT, map_location="cpu")); model.eval()
    recs = []
    with torch.no_grad():
        for r in rows:
            x = _EVAL_TRANSFORM(Image.open(r["image"]).convert("RGB")).unsqueeze(0)
            e = tok(cache[r["image"]], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            o = model(x, e["input_ids"], e["attention_mask"])
            recs.append(dict(image=r["image"], gold_level1=r["level1"], gold_level2=r["level2"],
                             pred_level1=LEVEL1_CLASSES[o["level1_logits"].argmax(1).item()],
                             pred_level2=LEVEL2_CLASSES[o["level2_logits"].argmax(1).item()]))
    json.dump(recs, open(f"{OUT}/tamil_test_predictions.json", "w"), indent=1)
    res = {"n": len(recs), "dataset": "Tamil POLITICAL memes (not misogyny set)",
           "ocr": "tesseract tam+eng", "ocr_nonempty": sum(bool(v) for v in cache.values())}
    train = list(csv.DictReader(open(TEST.replace("test.csv", "train.csv"))))
    for lv, classes in (("level1", LEVEL1_CLASSES), ("level2", LEVEL2_CLASSES)):
        g = [r[f"gold_{lv}"] for r in recs]; p = [r[f"pred_{lv}"] for r in recs]
        gl = [c for c in classes if c in set(g)]  # macro over classes present in Tamil gold
        maj = max(set(r[lv] for r in train), key=[r[lv] for r in train].count)
        res[lv] = dict(gold_classes=gl, macro_f1=f1_score(g, p, labels=gl, average="macro", zero_division=0),
                       ci95=boot_f1(g, p, gl),
                       majority_class=maj, majority_f1=f1_score(g, [maj] * len(g), labels=gl, average="macro", zero_division=0),
                       majority_ci95=boot_f1(g, [maj] * len(g), gl),
                       per_class_f1=dict(zip(gl, f1_score(g, p, labels=gl, average=None, zero_division=0).tolist())),
                       pred_counts={c: p.count(c) for c in classes}, gold_counts={c: g.count(c) for c in classes})
    json.dump(res, open(f"{OUT}/tamil_transfer_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
