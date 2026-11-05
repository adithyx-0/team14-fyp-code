# evaluation/

`python -m evaluation.aggregate` → `outputs/evaluation/consolidated.{md,json}`.
Inputs: per-example prediction files (records of `image, gold_level1, gold_level2, pred_level1, pred_level2`;
Gemini's CSV uses `true_*`/`pred_*`). Any `outputs/arm2_vlm_benchmark/*.json` or `outputs/arm3_peft_finetune/*.json`
in that format is picked up automatically; missing systems are listed as "not run". Arm 1's per-example file is
produced by `python -m arm1_classical.predict_test` (reproduces 0.5076 / 0.3940 exactly).

## Why these choices
- **Macro-F1, not accuracy.** Level-1 is ~95% troll in train (477/500), so a classifier that always says "troll"
  scores ~95% accuracy. Macro-F1 averages per-class F1 with equal weight, so the rare class (support) counts as
  much as the common one. It is also the metric SYNAPSE reports, which keeps the comparison like-for-like.
  The majority-class row shows the floor any real system has to clear.
- **UNKNOWN = wrong.** A malformed or failed generation is a real failure mode of a VLM arm; it is a miss, never dropped.
  The UNKNOWN rate is reported separately.
- **Bootstrap CIs.** The test set has only 100 examples (15 `intersection`, a handful of `support`), so a
  single macro-F1 has wide sampling noise. We resample the 100 test items with replacement (2000×, fixed seed),
  recompute macro-F1, and report the 2.5–97.5 percentile interval. No distributional assumption is needed.
- **Paired comparison.** The same resample indices are used for every system, so the difference Δ = system − Arm 1
  is computed on identical resamples; this removes test-item difficulty from the comparison and is far tighter than
  comparing two independent CIs. p = 2·min(P(Δ≤0), P(Δ≥0)). A CI on Δ containing 0 means the data do not separate
  the systems.
- **Gold-label check.** Every system's gold columns are checked against `test.csv`; a mismatch invalidates the row
  rather than silently scoring against the wrong labels.
- **Paper numbers** are shown as *reported, not reproduced*; they have no CI because we do not have per-example outputs.
