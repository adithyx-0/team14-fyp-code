# Explainability — Grad-CAM on Arm 1

Grad-CAM applied to Arm 1's frozen Swin-Tiny vision branch, using the
final-stage feature map (`forward_features()`, shape `[B,7,7,768]` — each
cell covers a 32x32px patch of the 224x224 input) as the CAM target layer,
the standard adaptation of Grad-CAM to Swin transformers in place of a
CNN's last conv layer. Since the backbone is frozen (`requires_grad=False`
on its weights), the input image tensor itself is marked
`requires_grad=True` so gradients still flow to the feature map even though
no weight gradients are computed. See `explainability/gradcam_arm1.py` for
the full method (docstring + implementation).

Run with:
```
python -m explainability.gradcam_arm1 --n 16 --level level1
```
(`--level level2` for the person/party/intersection heatmaps instead.)

## Sample outputs (`outputs/explainability/gradcam/`)

Level-1 and level-2 overlays for held-out test images, covering correct and
incorrect predictions; filenames encode
`{id}_{level}_true-{label}_pred-{label}_{correct|wrong}.png`. Per-example
`ExplanationRecord`s (shared schema, `explainability/schema.py`) are in
`records/` and `gradcam/records_level{1,2}.json`.

Qualitative reading (single examples, not evidence on their own): on
`2.jpg` (support, mispredicted troll) the hottest region sits on the two
people rather than the background; on `14.jpg` (troll, correct) the map is
diffuse. The quantitative test below is what supports any claim.

## Quantitative faithfulness: deletion / insertion AUC

Method (RISE; Petsiuk, Das & Saenko, BMVC 2018): remove (deletion) or reveal
(insertion) the 7x7 Grad-CAM patches in order of decreasing saliency and
track the model's probability for its predicted class. Faithful map =>
deletion AUC low, insertion AUC high. Control: the same curves with a random
patch order (3 seeds/image). Paired one-sided Wilcoxon signed-rank test over
images. Only the image branch is perturbed; OCR text is held fixed.
Grad-CAM: Selvaraju et al., ICCV 2017. Code: `explainability/faithfulness.py`,
raw numbers: `faithfulness_results.json`.

n = 30 test images (first 30 rows of the test set), 3 random seeds.

| Level | Deletion AUC Grad-CAM (lower=better) | Deletion random | Insertion AUC Grad-CAM (higher=better) | Insertion random | Paired p (del / ins) |
|---|---|---|---|---|---|
| 1 (troll/support) | 0.567 +/- 0.312 | 0.768 +/- 0.189 | 0.893 +/- 0.153 | 0.740 +/- 0.230 | 2.8e-05 / 3.0e-06 |
| 2 (person/party/intersection) | 0.311 +/- 0.169 | 0.516 +/- 0.189 | 0.611 +/- 0.288 | 0.496 +/- 0.283 | 8.2e-08 / 3.5e-05 |

Grad-CAM ordering beats random ordering on both metrics and both levels
(Grad-CAM better on 80-90% of images). So the heatmaps are faithful to the
model's behaviour at the resolution tested, beyond what chance ordering gives.

## Caveats to state up front

- n = 30 images, not the full 100: a CPU/MPS-bound run that shared a 16GB
  machine with training jobs. The effect is large and highly significant,
  but the sample is small; extend with `--limit 100` for the final report.
- 7x7 resolution is coarse (Swin-Tiny's last stage): maps localise to
  regions, not edges.
- Faithfulness here is with respect to the model, not human plausibility;
  no human-annotated ground-truth regions exist for this dataset.
- Image branch only: the text (OCR) branch is not explained, and the NL
  rationale generator for Arms 2/3 is not built, so a cross-modal
  faithfulness score between the two (Phase 5's last item) is still open.
