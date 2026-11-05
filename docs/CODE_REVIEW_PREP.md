# Code review prep (read-only review; two trivial fixes applied)

Reviewed: `arm2_vlm_benchmark/`, `arm3_peft_finetune/`, `evaluation/`,
`explainability/`, `demo/`. Ranked by how much they could matter in a review.

## Worth deciding (not changed)
1. **Bootstrap CI for Level 1 is slightly distorted** (`evaluation/aggregate.py`,
   `macro_f1_boot`). With 4 support images, about 1.7% of resamples
   (0.96^100) contain no support image, and the code scores that class as F1 = 0.
   This lowers the lower CI bound and widens the interval a little. Either
   exclude or flag those resamples, or state the convention in the README.
2. **Comparability claim in `outputs/arm1_classical/RESULTS.md`** says our test
   set is "the real held-out set that SYNAPSE's own numbers come from".
   That is not verified (see `DEVIATIONS.md`, "Level-1 comparability"). Soften
   the wording.
3. **Cross-directory dependency**: `evaluation/aggregate.py`,
   `arm2_vlm_benchmark/local_vlm.py` and `arm3_peft_finetune/run_ablation.py`
   read `../test fyp/` (models, resized images, replication predictions).
   Anyone who clones only this repo cannot reproduce Arm 2/3 or the
   replication row. Document it in the README or vendor the small inputs.
4. **Different image inputs across arms**: Arm 1 and Gemini read the original
   images; Qwen arms read the 512px resized copies. Both are documented, but the
   claim "same images" in comments is only true for ids and labels. Say so.
5. **Few-shot and other 7B items** are in the plan but absent; `DEVIATIONS.md`
   covers it, but make sure slides use the same wording.

## Minor / cosmetic (not changed; arm3 files are in use by a running job)
6. `arm3_peft_finetune/train_lora.py`: docstring says "Usage: python
   scripts/train_lora.py ... --out adapters/synapse_lora", which is the old
   replication path; `steps_per_eval` comment says "for the smoke test".
   Edit only after the background runs finish, since `run_ablation.py` launches
   this file as a subprocess for each run.
7. `arm2_vlm_benchmark/local_vlm.py`: model and image paths are relative
   (`../test fyp/...`), so it only works when run from `code/`. Fail early with
   a clear message, or anchor them to `__file__`.
8. `local_vlm.parse_output`: a regex takes the first word after `INTENT:`;
   if a model echoes the template ("INTENT: SUPPORT or CRITICISE") the first
   option wins. Harmless for CoT (last line wins) but worth a guard for
   zero-shot. Current UNKNOWN rate is 0%, so it has not bitten.
9. `gemini_zeroshot.py` hardcodes the Arm 1 and paper numbers in its output
   JSON; they can drift from `consolidated.md`. Low risk, but read them from one place.
10. `explainability/faithfulness.py` default `--out` is the same file the n=30
    run wrote, so running it with defaults overwrites that result. The n=100
    results went to `faithfulness_results_n100.json`. `RESULTS.md` still
    reports n=30; update it.
11. `demo/app.py` defaults the smoke test's temp file to `/tmp`; fine, but use
    the scratch dir convention if you want this tidy.

## Applied (trivial, safe)
- `explainability/faithfulness.py`: removed an unused variable (`cell`) in
  `patch_mask_stack`.
- `demo/app.py`: replaced a stale comment ("being refactored").

## Looks good
- Seeding, NaN-run rejection and resumable grid in `run_ablation.py`.
- Metrics are computed once, the same way, for every system in `aggregate.py`.
- Unparseable outputs are counted as wrong and reported, not dropped.

## Not reviewed
`arm1_classical/` training code was read only for the CV plan; the website
(`website/`) was reviewed when written, not re-reviewed here.
