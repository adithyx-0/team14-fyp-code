"""Hero chart: Avg-F1 (mean of L1,L2 macro-F1) with 95% bootstrap CIs for every
system in outputs/evaluation/consolidated.json. Writes outputs/figures/hero_avg_f1.{png,svg}.
SYNAPSE paper row = reported, not reproduced (grey, dashed, no CI available).
Usage: python -m evaluation.make_hero_chart
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "outputs/figures"
INK, MUTED, GREY = "#1f2937", "#6b7280", "#9ca3af"
C_BASE, C_ARM1, C_VLM, C_LORA, C_API = "#9ca3af", "#2a6fbb", "#e08a1e", "#2e8b57", "#7b3fa0"


def label(n):
    r = {"Majority-class baseline (train mode: troll/person)": "Majority-class baseline",
         "Arm 1 — Swin-Tiny + MuRIL (classical)": "Arm 1: Swin-Tiny + MuRIL",
         "SYNAPSE replication — Qwen3-VL-2B LoRA (mlx)": "SYNAPSE recipe replication (Qwen3-VL-2B LoRA)",
         "Gemini 3.5 Flash-Lite zero-shot": "Gemini 3.5 Flash-Lite zero-shot (API)",
         "arm2_vlm_benchmark/qwen3vl2b_cot": "Qwen3-VL-2B CoT",
         "arm2_vlm_benchmark/qwen3vl2b_zeroshot": "Qwen3-VL-2B zero-shot"}
    if n in r: return r[n]
    if n.startswith("arm3_peft_finetune/"):
        t = n.split("/")[1].replace("_os", " +oversample").replace("_s", " seed ")
        t = t.replace("r4 +", "r4 +")
        return "Arm 3 LoRA " + t.replace("r", "rank ", 1)
    if n.startswith("arm2_vlm_benchmark/"): return n.split("/")[1]
    return n


def color(n):
    if n.startswith("Majority"): return C_BASE
    if n.startswith("Arm 1"): return C_ARM1
    if n.startswith("arm3") or n.startswith("SYNAPSE rep"): return C_LORA
    if n.startswith("Gemini"): return C_API
    return C_VLM


def main():
    d = json.load(open("outputs/evaluation/consolidated.json"))
    rows = [(label(s["name"]), s["avg"], s["ci_avg"], color(s["name"])) for s in d["systems"] if s["status"] == "ok"]
    ref = d.get("reported_reference") or [{"name": "SYNAPSE paper", "f1_l1": 0.92, "f1_l2": 0.4256}]
    for r in (ref if isinstance(ref, list) else [ref]):
        if "SYNAPSE" in r.get("name", ""):
            rows.append(("SYNAPSE paper (reported, not reproduced)", (r["f1_l1"] + r["f1_l2"]) / 2, None, None))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=200)
    for i, (n, v, ci, c) in enumerate(rows):
        if ci is None:
            ax.barh(i, v, color="#e5e7eb", edgecolor=GREY, linestyle="--", linewidth=2, height=0.62, hatch="//")
            ax.text(v + 0.01, i, f"{v:.3f}  (no CI; split unverified)", va="center", fontsize=12, color=MUTED)
        else:
            ax.barh(i, v, color=c, height=0.62)
            ax.errorbar(v, i, xerr=[[v - ci[0]], [ci[1] - v]], color=INK, capsize=5, lw=2)
            ax.text(ci[1] + 0.01, i, f"{v:.3f}", va="center", fontsize=12, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=13, color=INK)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[2] is None: t.set_color(MUTED); t.set_style("italic")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Avg-F1 = (Level-1 macro-F1 + Level-2 macro-F1) / 2", fontsize=14, color=INK)
    ax.tick_params(axis="x", labelsize=12)
    ax.xaxis.grid(True, color="#e5e7eb"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("Malayalam political-meme intent and target: Avg-F1 by system", fontsize=18, fontweight="bold", color=INK, loc="left", pad=16)
    ax.legend(handles=[Patch(color=C_BASE, label="Baseline"), Patch(color=C_ARM1, label="Arm 1 (classical)"),
                       Patch(color=C_VLM, label="Arm 2 local VLM, prompt-only"), Patch(color=C_API, label="Arm 2 API model, prompt-only"),
                       Patch(color=C_LORA, label="Arm 3 LoRA / replication"),
                       Patch(facecolor="#e5e7eb", edgecolor=GREY, linestyle="--", hatch="//", label="Paper (reported only)")],
              loc="lower right", fontsize=12, frameon=False)
    fig.text(0.01, 0.01, "n = 100 held-out test images (96 troll / 4 support at L1). Bars: point estimate; whiskers: 95% percentile bootstrap CI (2000 resamples).\n"
             "L1 rests on only 4 support images, so CIs are wide. SYNAPSE paper value is quoted from our notes; its split is unverified.",
             fontsize=11, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"): fig.savefig(f"{OUT}/hero_avg_f1.{ext}")
    print("saved")


if __name__ == "__main__":
    main()
