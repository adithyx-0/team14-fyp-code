# Pooled repeated CV — Arm 1 on all 600 labelled images

Train+test pooled (600 images, 27 support). Stratified 5-fold on level 1,
10 repeats (50 head trainings per system). Frozen Swin-Tiny/MuRIL features
cached once (`cache_features.py`); heads trained with Arm 1's exact recipe
(8 epochs, batch 16, AdamW 1e-3, class-weighted CE, random flip), fixed
epochs, **no epoch selection on the validation fold**. Macro-F1 on pooled
out-of-fold predictions; ± is std over the 10 repeats; CI is a 2000-resample
bootstrap over images (repeat-averaged). Raw numbers: `outputs/cv_pooled/cv_results.json`.

Cache equivalence: Arm 1's saved fold-0 checkpoint scored on cached features
gives L1 0.5116 / L2 0.4124, identical to its stored CV result (`verify_cache.py`).

| System | L1 macro-F1 [95% CI] | L2 macro-F1 [95% CI] |
|---|---|---|
| Majority class | 0.488 ± 0.000 [0.484, 0.492] | 0.259 ± 0.000 [0.250, 0.268] |
| Text only (MuRIL) | 0.468 ± 0.030 [0.460, 0.475] | 0.285 ± 0.033 [0.274, 0.295] |
| Image only (Swin) | 0.566 ± 0.025 [0.525, 0.604] | 0.405 ± 0.026 [0.377, 0.430] |
| Full (image+text) | 0.565 ± 0.018 [0.524, 0.603] | 0.405 ± 0.020 [0.379, 0.430] |

Paired bootstrap (same resamples): full − majority is significant at both levels (L1 p<0.001, L2 p<0.001; L2 Δ +0.146 [0.116, 0.175]). Full − image-only: Δ −0.001 (L1) and +0.001 (L2), CIs contain 0 (p≈0.9). Full − text-only: Δ +0.096 (L1), +0.120 (L2), p<0.001.

Findings:
1. With 600 images the Arm 1 gain over the majority baseline is statistically solid (it was not at n=100 for L1).
2. **The OCR text branch adds nothing measurable**: image-only equals the fused model. Text alone is at or below majority for L1. The model is effectively image-driven. Caveat (independent check): the frozen MuRIL [CLS] features are nearly constant across samples (per-dim std ~0.0008, unstandardised) and 109 of 500 train OCR texts are empty, so this is a statement about Arm 1's frozen-[CLS] text pathway, not proof that text is uninformative. Next step: standardise features or use mean-pooled tokens and re-run (cheap on cached features).
3. Official-split seed variance of Arm 1 (10 seeds, train 500 → test 100): L1 0.580 ± 0.059, L2 0.374 ± 0.054 (ranges 0.48–0.66 and 0.29–0.46). The single published run (0.508 / 0.394) sits inside this spread; differences of ~0.1 between single-seed systems on the 100-image test are noise.

4. **Standardised-text re-run** (`python cv_pooled/run_cv.py --std-text`, raw numbers `outputs/cv_pooled/cv_results_stdtext.json`): text features z-scored with each training fold's mean/std. Text-only rises above majority (L1 0.509 [0.480, 0.543], L2 0.347 [0.324, 0.368]), but full vs image-only is still null: L1 0.548 vs 0.566 (Δ -0.018, p=0.43), L2 0.401 vs 0.405 (Δ -0.003, p=0.82). Conclusion: with a frozen MuRIL [CLS] pathway, text adds no measurable value over the image even after fixing the scale problem. Mean-pooled tokens or a fine-tuned text encoder were not tested.

Caveats: CV numbers are not held-out test numbers and are not comparable to the SYNAPSE paper's reported scores. Hyperparameters were not tuned on these folds. Only the frozen-feature heads are cross-validated; the VLM arms (Arm 2/3) are not (cost, see `docs/CV_PLAN.md`).
