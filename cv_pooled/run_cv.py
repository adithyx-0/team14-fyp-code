"""Repeated stratified 5-fold CV on the pooled 600 labelled images, using
cached frozen features (see docs/CV_PLAN.md). Heads are trained for a fixed
8 epochs; no epoch is selected on the validation fold.

Systems: full (image+text), image-only, text-only, majority-class.
Usage: python cv_pooled/run_cv.py [--repeats 10]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm1_classical.model import ResidualFusionBlock  # noqa: E402

FEATS = "outputs/cv_pooled/features.npz"
OUT = "outputs/cv_pooled"
L1 = ["troll", "support"]
L2 = ["person", "party", "intersection"]
EPOCHS, BATCH, LR = 8, 16, 1e-3  # identical to arm1_classical/train.py
STD_TEXT = False  # set by --std-text: z-score text features with train-fold stats


class Head(nn.Module):
    """Same layers as ClassicalBaseline minus the frozen backbones."""

    def __init__(self, dim=512):
        super().__init__()
        self.vp, self.tp = nn.Linear(768, dim), nn.Linear(768, dim)
        self.blocks = nn.ModuleList([ResidualFusionBlock(dim) for _ in range(2)])
        self.h1, self.h2 = nn.Linear(dim, 2), nn.Linear(dim, 3)

    def forward(self, img, txt):
        x = self.vp(img) + self.tp(txt)
        for b in self.blocks:
            x = b(x)
        return self.h1(x), self.h2(x)


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return f.mean()


def weights(y, k):
    c = np.maximum(np.bincount(y, minlength=k), 1)
    return torch.tensor(c.sum() / (k * c), dtype=torch.float32)


def train_predict(tr, te, d, mode, seed):
    """Train head on rows `tr`, return (l1_pred, l2_pred) for rows `te`."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    img = torch.tensor(d["img"]); flip = torch.tensor(d["img_flip"]); txt = torch.tensor(d["txt"])
    if STD_TEXT:
        mu, sd = txt[tr].mean(0, keepdim=True), txt[tr].std(0, keepdim=True)
        txt = (txt - mu) / (sd + 1e-6)
    if mode == "image_only":
        txt = torch.zeros_like(txt)
    if mode == "text_only":
        img, flip = torch.zeros_like(img), torch.zeros_like(flip)
    y1, y2 = d["y1"], d["y2"]
    m = Head()
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    c1 = nn.CrossEntropyLoss(weight=weights(y1[tr], 2))
    c2 = nn.CrossEntropyLoss(weight=weights(y2[tr], 3))
    t1, t2 = torch.tensor(y1), torch.tensor(y2)
    for _ in range(EPOCHS):
        m.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            use_flip = torch.tensor(rng.rand(len(b)) < 0.5).unsqueeze(1)
            x = torch.where(use_flip, flip[b], img[b])
            o1, o2 = m(x, txt[b])
            loss = c1(o1, t1[b]) + c2(o2, t2[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        o1, o2 = m(img[te], txt[te])
    return o1.argmax(-1).numpy(), o2.argmax(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--std-text", action="store_true",
                    help="standardise text features per training fold; writes cv_results_stdtext.json")
    a = ap.parse_args()
    global STD_TEXT
    STD_TEXT = a.std_text
    z = np.load(FEATS)
    d = {k: z[k] for k in ("img", "img_flip", "txt")}
    d["y1"] = np.array([L1.index(v) for v in z["level1"]])
    d["y2"] = np.array([L2.index(v) for v in z["level2"]])
    n = len(d["y1"])
    systems = ["full", "image_only", "text_only"]
    oof = {s: np.zeros((a.repeats, 2, n), dtype=int) for s in systems}
    maj = (np.bincount(d["y1"]).argmax(), np.bincount(d["y2"]).argmax())
    for r in range(a.repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=1000 + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), d["y1"])):
            for s in systems:
                p1, p2 = train_predict(tr, te, d, s, seed=r * 10 + f)
                oof[s][r, 0, te], oof[s][r, 1, te] = p1, p2
        print("repeat", r, "done", flush=True)
    oof["majority"] = np.stack([np.stack([np.full(n, maj[0]), np.full(n, maj[1])])] * a.repeats)

    # per-repeat pooled OOF macro-F1 + bootstrap over images (repeat-averaged)
    res, B = {}, 2000
    br = np.random.RandomState(42)
    idx = br.randint(0, n, (B, n))
    boot = {}
    for s, o in oof.items():
        per = np.array([[macro_f1(d["y1"], o[r, 0], 2), macro_f1(d["y2"], o[r, 1], 3)]
                        for r in range(a.repeats)])
        bs = np.array([[np.mean([macro_f1(d[y][ix], o[r, l][ix], k) for r in range(a.repeats)])
                        for l, (y, k) in enumerate((("y1", 2), ("y2", 3)))] for ix in idx])
        boot[s] = bs
        res[s] = {"mean": per.mean(0).tolist(), "std_over_repeats": per.std(0).tolist(),
                  "ci95_L1": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                  "ci95_L2": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                  "per_repeat": per.tolist()}
    # paired differences on the same resamples
    pairs = {}
    for a_, b_ in (("full", "majority"), ("full", "image_only"), ("full", "text_only"),
                   ("image_only", "majority"), ("text_only", "majority")):
        dd = boot[a_] - boot[b_]
        pairs[f"{a_} - {b_}"] = {
            f"L{l+1}": {"delta": float(dd[:, l].mean()),
                        "ci95": np.percentile(dd[:, l], [2.5, 97.5]).tolist(),
                        "p": float(min(1, 2 * min((dd[:, l] <= 0).mean(), (dd[:, l] >= 0).mean())))}
            for l in (0, 1)}
    # per-class F1 (repeat-averaged) for full
    pc = {}
    for s in ("full", "image_only", "text_only"):
        for l, (y, k, names) in enumerate((("y1", 2, L1), ("y2", 3, L2))):
            for c, nm in enumerate(names):
                fs = []
                for r in range(a.repeats):
                    p = oof[s][r, l]
                    tp = ((p == c) & (d[y] == c)).sum(); fp = ((p == c) & (d[y] != c)).sum()
                    fn = ((p != c) & (d[y] == c)).sum()
                    fs.append(2 * tp / max(2 * tp + fp + fn, 1))
                pc.setdefault(s, {})[nm] = float(np.mean(fs))
    # seed variance on the official split (train rows -> test rows)
    tr = np.where(z["split"] == "train")[0]; te = np.where(z["split"] == "test")[0]
    seeds = []
    for sd in range(10):
        p1, p2 = train_predict(tr, te, d, "full", seed=500 + sd)
        seeds.append([macro_f1(d["y1"][te], p1, 2), macro_f1(d["y2"][te], p2, 3)])
    seeds = np.array(seeds)
    out = {"n_images": n, "repeats": a.repeats, "support_count": int((d["y1"] == 1).sum()),
           "systems": res, "paired": pairs, "per_class_f1": pc,
           "official_split_10_seeds": {"mean": seeds.mean(0).tolist(), "std": seeds.std(0).tolist(),
                                       "min": seeds.min(0).tolist(), "max": seeds.max(0).tolist()}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/cv_results{'_stdtext' if STD_TEXT else ''}.json", "w"), indent=2)
    print(json.dumps({s: {k: v for k, v in r.items() if k != "per_repeat"} for s, r in res.items()}, indent=1))
    print(json.dumps(pairs, indent=1)); print(pc); print(out["official_split_10_seeds"])


if __name__ == "__main__":
    main()
