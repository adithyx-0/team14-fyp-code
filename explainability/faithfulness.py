"""Quantitative faithfulness of Arm 1's Grad-CAM: deletion / insertion AUC.

Metric (Petsiuk, Das & Saenko, "RISE: Randomized Input Sampling for
Explanation of Black-box Models", BMVC 2018):
  - Deletion: starting from the real image, remove (replace with a neutral
    baseline) the image patches in order of decreasing saliency and track the
    model's probability for the originally predicted class. A faithful map
    makes the probability collapse fast -> LOW area under the curve (AUC).
  - Insertion: starting from a blurred image, reveal patches in order of
    decreasing saliency. A faithful map makes the probability recover fast ->
    HIGH AUC.
Patches are the 7x7 Grad-CAM cells (each 32x32 px of the 224x224 input), so
the perturbation granularity equals the explanation's own resolution (49
steps, 50 points per curve). x-axis = fraction of patches removed/inserted in
[0, 1]; AUC by trapezoid rule, so values live in [0, 1] (max possible = prob).

Control: the same curves with a RANDOM patch order (N_SEEDS seeds, averaged
per image). Grad-CAM only counts as faithful here if it beats random, tested
with a paired Wilcoxon signed-rank test across images.

Perturbation baselines follow RISE: deletion -> zero in normalized space
(= ImageNet mean colour); insertion -> Gaussian-blurred copy of the image.

Only the IMAGE branch is perturbed; OCR text is held fixed. If the model
leans mostly on text, both curves are flat and the metric cannot distinguish
Grad-CAM from random -- reported as such, not hidden.

Usage:
    python -m explainability.faithfulness [--limit 100] [--seeds 5] [--device mps]
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import wilcoxon

from explainability.gradcam_arm1 import (
    OUT_DIR as _GRADCAM_DIR,
    TEST_OCR_CSV,
    compute_gradcam_full,
    get_device,
    load_model,
    prepare_row,
)
from explainability.schema import ExplanationRecord, ImageRegions, save_records

GRID = 7
PATCH = 32  # 224 / 7
OUT_ROOT = "outputs/explainability"
BLUR_KERNEL, BLUR_SIGMA = 51, 15.0


def patch_mask_stack(order):
    """order: permutation of range(49) (most salient first).
    Returns bool tensor [50, 1, 224, 224]; row k = True where the first k
    patches of `order` are selected."""
    masks = [torch.zeros(1, 224, 224, dtype=torch.bool)]
    sel = torch.zeros(GRID * GRID, dtype=torch.bool)
    for idx in order:
        sel = sel.clone()
        sel[idx] = True
        m = sel.view(GRID, GRID).repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)
        masks.append(m.unsqueeze(0))
    return torch.stack(masks)  # [50, 1, 224, 224]


@torch.no_grad()
def class_prob(model, imgs, text_proj, level, pred_idx):
    """Softmax prob of `pred_idx` for a batch of images with FIXED text."""
    feats = model.vision.forward_features(imgs)  # [B,7,7,768]
    fused = model.vision_proj(feats.mean(dim=(1, 2))) + text_proj
    for block in model.fusion_blocks:
        fused = block(fused)
    head = model.level1_head if level == "level1" else model.level2_head
    return torch.softmax(head(fused), dim=1)[:, pred_idx]


def auc(y):
    """Trapezoid area under y sampled at len(y) equally spaced points on [0,1]."""
    y = np.asarray(y, dtype=np.float64)
    return float(np.trapezoid(y, dx=1.0 / (len(y) - 1)))


def curves(model, img_t, text_proj, level, pred_idx, orders, blurred, device, chunk=150):
    """Deletion and insertion probability curves for several patch orders at
    once (one big batched pass instead of one small pass per order, which is
    far faster when the accelerator is shared). Returns (del, ins), each
    [len(orders), 50]."""
    masks = torch.cat([patch_mask_stack(o) for o in orders]).to(device)  # [n*50,1,H,W]
    img = img_t.to(device).unsqueeze(0)
    probs = {"del": [], "ins": []}
    for start in range(0, masks.shape[0], chunk):
        m = masks[start : start + chunk]
        # deletion: selected (most-salient-first) patches replaced by 0 (mean colour)
        probs["del"].append(class_prob(model, torch.where(m, torch.zeros_like(img), img), text_proj, level, pred_idx))
        # insertion: selected patches revealed from the real image over a blurred base
        probs["ins"].append(class_prob(model, torch.where(m, img, blurred), text_proj, level, pred_idx))
    n = len(orders)
    d = torch.cat(probs["del"]).cpu().numpy().reshape(n, -1)
    i = torch.cat(probs["ins"]).cpu().numpy().reshape(n, -1)
    return d, i


def cam_order(cam, raw):
    """Most-salient-first patch order. Ties among ReLU-zeroed cells are broken
    by the pre-ReLU Grad-CAM value (stable, deterministic)."""
    key = np.lexsort((-raw.reshape(-1), -cam.reshape(-1)))
    return key.tolist()


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}


def run_level(model, tokenizer, rows, level, seeds, device):
    per_image, records = [], []
    t_start = time.time()
    for k, row in enumerate(rows):
        if k % 5 == 0:
            print(f"  [{level}] {k}/{len(rows)} images, {time.time() - t_start:.0f}s elapsed", flush=True)
        img_t, input_ids, attention_mask = prepare_row(row, tokenizer)
        g = compute_gradcam_full(model, img_t, input_ids, attention_mask, device, level=level)
        pred_idx = g["pred_idx"]
        pred_label = g["class_names"][pred_idx]

        with torch.no_grad():
            text_out = model.text(
                input_ids=input_ids.unsqueeze(0).to(device),
                attention_mask=attention_mask.unsqueeze(0).to(device),
            )
            text_proj = model.text_proj(text_out.last_hidden_state[:, 0, :])
            blurred = TF.gaussian_blur(img_t.unsqueeze(0), BLUR_KERNEL, BLUR_SIGMA).to(device)

        orders = [cam_order(g["cam"], g["raw"])]
        for sd in range(seeds):
            orders.append(np.random.default_rng(1000 * k + sd).permutation(GRID * GRID).tolist())
        D, I = curves(model, img_t, text_proj, level, pred_idx, orders, blurred, device)
        d_g, i_g = D[0], I[0]
        d_r = [auc(c) for c in D[1:]]
        i_r = [auc(c) for c in I[1:]]

        item = {
            "image": row["image"],
            "pred": pred_label,
            "true": row[level],
            "p_pred": g["prob"],
            "cam_all_zero": bool(g["cam"].max() < 1e-6),
            "del_auc_gradcam": auc(d_g),
            "ins_auc_gradcam": auc(i_g),
            "del_auc_random": float(np.mean(d_r)),
            "ins_auc_random": float(np.mean(i_r)),
            "del_curve_gradcam": d_g.tolist(),
            "ins_curve_gradcam": i_g.tolist(),
        }
        per_image.append(item)
        records.append(
            ExplanationRecord(
                example_id=os.path.splitext(os.path.basename(row["image"]))[0],
                arm="arm1",
                level=level,
                pred=pred_label,
                true_label=row[level],
                image_regions=ImageRegions(heatmap=g["cam"].tolist(), method="gradcam"),
                faithfulness={
                    "deletion_auc": item["del_auc_gradcam"],
                    "insertion_auc": item["ins_auc_gradcam"],
                    "deletion_auc_random": item["del_auc_random"],
                    "insertion_auc_random": item["ins_auc_random"],
                },
            )
        )
    return per_image, records


def aggregate(per_image):
    out = {}
    for key in ("del_auc_gradcam", "del_auc_random", "ins_auc_gradcam", "ins_auc_random"):
        out[key] = summarize([p[key] for p in per_image])
    for metric in ("del", "ins"):
        diff = np.array([p[f"{metric}_auc_gradcam"] - p[f"{metric}_auc_random"] for p in per_image])
        entry = {"mean_diff_gradcam_minus_random": float(diff.mean())}
        # "better" = lower for deletion, higher for insertion
        better = (diff < 0) if metric == "del" else (diff > 0)
        entry["frac_images_gradcam_better"] = float(better.mean())
        try:
            alt = "less" if metric == "del" else "greater"
            entry["wilcoxon_p_one_sided"] = float(wilcoxon(diff, alternative=alt).pvalue)
        except ValueError:  # all differences zero
            entry["wilcoxon_p_one_sided"] = None
        out[f"{metric}_paired_test"] = entry
    out["n_cam_all_zero"] = int(sum(p["cam_all_zero"] for p in per_image))
    out["mean_p_pred"] = float(np.mean([p["p_pred"] for p in per_image]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="number of test images (first N rows)")
    ap.add_argument("--seeds", type=int, default=5, help="random-order control seeds per image")
    ap.add_argument("--levels", nargs="+", default=["level1", "level2"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "faithfulness_results.json"))
    args = ap.parse_args()

    device = get_device(args.device)
    print("device:", device)
    with open(TEST_OCR_CSV) as f:
        rows = list(csv.DictReader(f))[: args.limit]

    model, tokenizer = load_model(device)
    results = {
        "metric": "RISE deletion/insertion AUC on 7x7 Grad-CAM patches (Arm 1, image branch only)",
        "n_images": len(rows),
        "random_seeds": args.seeds,
        "steps_per_curve": GRID * GRID + 1,
        "levels": {},
    }
    os.makedirs(os.path.join(OUT_ROOT, "records"), exist_ok=True)
    for level in args.levels:
        t0 = time.time()
        per_image, records = run_level(model, tokenizer, rows, level, args.seeds, device)
        results["levels"][level] = {"summary": aggregate(per_image), "per_image": per_image}
        save_records(records, os.path.join(OUT_ROOT, "records", f"faithfulness_{level}.json"))
        s = results["levels"][level]["summary"]
        print(f"[{level}] {time.time() - t0:.0f}s  n={len(per_image)}")
        for k in ("del_auc_gradcam", "del_auc_random", "ins_auc_gradcam", "ins_auc_random"):
            print(f"  {k}: {s[k]['mean']:.4f} +/- {s[k]['std']:.4f}")
        print(f"  del paired: {s['del_paired_test']}")
        print(f"  ins paired: {s['ins_paired_test']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
