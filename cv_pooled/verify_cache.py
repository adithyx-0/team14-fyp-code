"""Equivalence check: Arm 1's saved fold-0 checkpoint, run on CACHED features
for its validation fold, must reproduce the macro-F1 in outputs/arm1_classical/cv_results.json."""
import csv, json, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv_pooled.run_cv import Head, macro_f1, L1, L2
z = np.load("outputs/cv_pooled/features.npz")
pos = {k: i for i, k in enumerate(z["image"])}
rows = [r for r in csv.DictReader(open("data/splits/malayalam_train_folds.csv")) if r["fold"] == "0"]
ix = [pos[r["image"]] for r in rows]
sd = torch.load("outputs/arm1_classical/fold0_best.pt", map_location="cpu")
m = Head()
m.vp.load_state_dict({"weight": sd["vision_proj.weight"], "bias": sd["vision_proj.bias"]})
m.tp.load_state_dict({"weight": sd["text_proj.weight"], "bias": sd["text_proj.bias"]})
m.h1.load_state_dict({"weight": sd["level1_head.weight"], "bias": sd["level1_head.bias"]})
m.h2.load_state_dict({"weight": sd["level2_head.weight"], "bias": sd["level2_head.bias"]})
m.blocks.load_state_dict({k[len("fusion_blocks."):]: v for k, v in sd.items() if k.startswith("fusion_blocks.")})
m.eval()
with torch.no_grad():
    o1, o2 = m(torch.tensor(z["img"][ix]), torch.tensor(z["txt"][ix]))
y1 = np.array([L1.index(z["level1"][i]) for i in ix]); y2 = np.array([L2.index(z["level2"][i]) for i in ix])
print("cached  L1 %.4f L2 %.4f" % (macro_f1(y1, o1.argmax(-1).numpy(), 2), macro_f1(y2, o2.argmax(-1).numpy(), 3)))
print("stored ", json.load(open("outputs/arm1_classical/cv_results.json"))["fold0"])
