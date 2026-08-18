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
python -m explainability.gradcam_arm1 --n 16 --level level2
```

## Sample outputs (`outputs/explainability/gradcam/`)

**Level1** (troll/support): 8 held-out test images, spread across both
classes and both correct/incorrect predictions.

Visual spot-check: on `2.jpg` (support meme, mispredicted as troll), the
heatmap's hottest region sits on the two people in the image rather than
background/text — plausible attention. On `14.jpg` (troll, correctly
predicted), the heatmap is comparatively diffuse across the frame.

**Level2** (person/party/intersection target): 16 held-out test images,
spread across the three classes — 8 correct, 8 wrong. Filenames encode
`{id}_{level}_true-{label}_pred-{label}_{correct|wrong}.png`.

Four picked for the slide (one clean correct per class type, two
instructive wrong ones — a random 4-of-16 would have been weaker):

- `2_level2_true-person_pred-person_correct.png` — heat lands squarely on
  the two people's faces; clean, unambiguous case of the model attending
  to the right region for a person-target prediction.
- `28_level2_true-party_pred-party_correct.png` — heat spans both a
  face and the "BJP" party-emblem text overlay, not just the person —
  a good example of the model using party-identifying visual evidence
  distinct from the person-only case above.
- `35_level2_true-intersection_pred-person_wrong.png` — true label is
  intersection (both a person and a party target present), model's top-1
  pick was person only. The heatmap itself actually spans *both* faces in
  the image, i.e. the visual evidence supports "intersection" — the
  miss looks like a single-label classification-head limitation (forced
  top-1 pick) rather than the vision backbone attending to the wrong
  place. Good talking point for viva: heatmap plausibility and prediction
  correctness are separable failure modes.
- `41_level2_true-party_pred-intersection_wrong.png` — a 3x4 grid
  collage of individual portrait photos (a common meme format in this
  dataset); heat concentrates on one face in the top-left. True label is
  party, predicted intersection — plausible confusion, since a multi-face
  collage visually resembles the "multiple distinct targets" pattern that
  defines intersection, even when the intended reading is "this whole
  slate belongs to one party."

No formal faithfulness scoring against a rationale generator yet (that's
the NL-rationale half of Phase 5 in instructions.md, not built) — this is
the image-heatmap half only, which instructions.md §6 Phase 5 and the Mid
Review status note both flag as a legitimate partial/stretch deliverable
on its own.

## Caveats to state up front

- 7x7 spatial resolution is coarse (Swin-Tiny's last stage, not a
  high-res conv map) — heatmaps localize to quadrants/regions, not
  fine-grained edges.
- Both `level1` and `level2` heatmaps now generated and reviewed (8 and 16
  samples respectively).
- No quantitative faithfulness metric yet — this is qualitative
  visual-inspection evidence, not a scored deliverable.
- The `level2` wrong-prediction examples above illustrate a real nuance
  worth stating proactively in viva: a plausible/well-localized heatmap
  does not guarantee a correct label, and vice versa — Grad-CAM explains
  where the model looked, not whether its single-label decision was
  right. `35.jpg` in particular shows the heatmap capturing the correct
  (multi-target) visual evidence even though the classifier's forced
  top-1 output missed it.
