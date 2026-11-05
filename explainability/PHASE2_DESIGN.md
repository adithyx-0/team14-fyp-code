# Phase 2 explainability design (writing only, nothing implemented)

Two additions on top of the existing Grad-CAM + deletion/insertion work:
(a) a natural-language rationale generator, and (b) a cross-modal faithfulness
score between the image evidence (Grad-CAM heatmap) and the text rationale.
Both reuse `explainability/schema.py` (`ExplanationRecord`), which already has
the slots: `image_regions`, `salient_tokens`, `rationale`, `faithfulness` (dict).
No schema change is required for the first version.

## (a) Rationale generator

**Purpose.** For a meme and a predicted label, produce a short human-readable
reason ("the meme shows <person> with a mocking caption, so CRITICISE / PERSON").

**Which system generates it.** The VLM arms (Arm 2 and Arm 3 Qwen3-VL-2B). Arm 1
has no generator, so it keeps only the Grad-CAM channel. A rationale is
generated *after* the label, conditioned on it, so it explains the prediction
rather than steering it.

**Input.**
- meme image (same 512px resized copy used for inference),
- the model's own predicted `level1` and `level2` labels,
- a fixed instruction asking for at most 2 sentences that name (i) who or what
  is shown and (ii) the visible or textual cue for the tone.

**Output.** A string stored in `ExplanationRecord.rationale`, plus the raw
generation text kept in a log. Greedy decoding, token cap about 80, same
seeds/settings logged per run.

**Two risks to design around.**
1. Post-hoc rationales can sound plausible without reflecting the model's
   actual computation. This is exactly why (b) exists; do not present rationales
   as faithful until (b) says so.
2. The 2B model degenerates into repetition on free text (observed in the CoT
   smoke test, see `arm2_vlm_benchmark/local_vlm.py`). Keep the cap, detect
   repetition, and count unusable rationales instead of dropping them.

**First implementation (small).** Reuse the `local_vlm.py` loading and
generation path, add a `rationale` strategy in a new `explainability/rationale.py`,
run on the same 100 test images for Arm 2 zero-shot predictions, write
`outputs/explainability/rationales_arm2.json` as `ExplanationRecord`s.
Estimated about 30 minutes of compute on this Mac; needs the Mac otherwise idle.

## (b) Cross-modal faithfulness score

**Question.** Does the rationale talk about the same evidence the model's image
saliency points at? Image evidence = Grad-CAM heatmap (Arm 1) or a VLM
saliency map if one is added later. Text evidence = what the rationale mentions.

**Inputs.**
- `image_regions.heatmap` (7x7 grid in [0,1]) for the same meme,
- `rationale` string,
- the meme image.

**Method (proposed, simplest defensible version).**
1. Pull the entities the rationale refers to (person, party symbol, text
   banner). Use a short fixed list of region cues rather than a free parser:
   ask the same VLM, or a grounding model, for a coarse bounding box per
   mentioned entity. Where a box cannot be obtained, mark the entity
   "ungrounded" and report the ungrounded rate.
2. Convert each box to the 7x7 grid mask.
3. Score = fraction of heatmap mass inside the union of grounded boxes,
   `sum(heatmap * mask) / sum(heatmap)`, compared with the same quantity under
   a random box of equal area (permutation control, as in the existing
   deletion/insertion tests).

**Output.** `ExplanationRecord.faithfulness["crossmodal_overlap"]`,
`["crossmodal_overlap_random"]`, `["n_ungrounded"]`. Aggregated per arm and
level with the same paired Wilcoxon design as `faithfulness.py`.

**Metric definition for the report.** Overlap above random chance, averaged
over images, with the ungrounded fraction reported next to it so a high score
on a few groundable images is not oversold.

**Known limits, state up front.**
- Needs a grounding step that does not exist yet, and the 2B model's boxes may
  be poor. A weaker fallback is OCR-token overlap: does the rationale quote
  words that OCR found inside high-saliency cells?
- Arm 1 Grad-CAM is 7x7, so overlap is coarse.
- Heatmaps exist only for Arm 1; rationales only for Arm 2/3 (different models).
  Until VLM saliency is added, this score compares *Arm 1's* image evidence to
  *another model's* rationale, which is a weaker claim than "this model's
  explanation is self-consistent". Say so.

## Open decisions for the team
1. Grounding source: the VLM itself, a separate detector, or the OCR-overlap fallback.
2. Whether to add a VLM saliency method (attention rollout or gradient on the
   vision tower) so both channels come from one model.
3. Whether rationale quality also gets a small human check (e.g. 30 samples
   rated by two team members), since automatic scores will be noisy.

## Order of work
1. Rationale generator on Arm 2 outputs (small, self-contained).
2. OCR-overlap faithfulness (no new model needed).
3. Box-grounded overlap, only if time remains.
