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
python -m explainability.gradcam_arm1 --n 12 --level level1
```
(`--level level2` for the person/party/intersection heatmaps instead.)

## Sample outputs (`outputs/explainability/gradcam/`)

8 held-out test images run so far, spread across both `level1` classes and
both correct/incorrect predictions — filenames encode
`{id}_{level}_true-{label}_pred-{label}_{correct|wrong}.png`.

Visual spot-check: on `2.jpg` (support meme, mispredicted as troll), the
heatmap's hottest region sits on the two people in the image rather than
background/text — plausible attention. On `14.jpg` (troll, correctly
predicted), the heatmap is comparatively diffuse across the frame. No
formal faithfulness scoring against a rationale generator yet (that's the
NL-rationale half of Phase 5 in instructions.md, not built) — this is the
image-heatmap half only, which instructions.md §6 Phase 5 and the Mid
Review status note both flag as a legitimate partial/stretch deliverable
on its own.

## Caveats to state up front

- 7x7 spatial resolution is coarse (Swin-Tiny's last stage, not a
  high-res conv map) — heatmaps localize to quadrants/regions, not
  fine-grained edges.
- Only run on `level1` (troll/support) so far; `level2` (person/party/
  intersection) heatmaps are one flag away (`--level level2`) but not yet
  generated or reviewed.
- No quantitative faithfulness metric yet — this is qualitative
  visual-inspection evidence, not a scored deliverable.
