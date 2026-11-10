# Worker B notes (10 Nov 2026)

## Task 1 PaliGemma2-3B-mix-448 (8-bit MLX, mlx-community/paligemma2-3b-mix-448-8bit) zero-shot, n=100
Script: arm2_vlm_benchmark/paligemma_zeroshot.py (imports parser from local_vlm.py, unchanged). Output: outputs/paligemma/paligemma2_3b_zeroshot.json (aggregate.py schema, 512px images), scores: paligemma_scores.json.
Prompt changes (3 attempts, smoke test on first 3 images, parseability only):
- v1 (same text as Qwen zero-shot): all 3 = "Sorry, as a base VLM I am not trained to answer this question."
- v2 (two short questions, one call each): TARGET parsed, INTENT = "unanswerable".
- v3 (INTENT asked as "praise or mock", mapped praise->SUPPORT, mock->CRITICISE; TARGET as v2): used for full run. Deviation: two calls per image, not the single-format prompt.
Results (UNKNOWN counted wrong, 2000-resample bootstrap): L1 0.420 [0.225,0.585], L2 0.334 [0.257,0.418], Avg 0.377. UNKNOWN: L1 62/100, L2 13/100. L1 preds: 27 troll, 11 support, 62 unknown; L2: 57 person, 29 intersection, 1 party. Conclusion: PaliGemma-mix cannot do this task usefully zero-shot; near/below majority baseline. Not run through aggregate.py (not in its glob; copy/point it at outputs/paligemma/ to ingest). 1 build tried (download worked).

## Task 2 Tamil political memes cross-lingual (Arm 1 Malayalam checkpoint, read-only)
Script: cross_lingual/tamil_transfer.py; outputs/cross_lingual/{tamil_transfer_results.json,tamil_test_predictions.json,tamil_ocr_cache.json}.
Tesseract has `tam`; OCR = tam+eng (168/201 non-empty). Tamil labels: L1 troll/support, L2 person/party only (NO intersection class in Tamil; 48/201 predictions were "intersection" and count as errors). Macro-F1 computed over classes present in Tamil gold (L2: person, party). Label mismatch is the only one; all labels mapped explicitly, unknowns raise.
n=201 (175 troll/26 support; 159 person/42 party). L1 0.474 [0.416,0.539] vs majority(troll) 0.465 [0.451,0.478]; per-class troll 0.827, support 0.121. L2 0.455 [0.382,0.527] vs majority(person) 0.442 [0.422,0.458]; per-class person 0.646, party 0.264. => no meaningful transfer beyond majority baseline. Note Tamil train majority used for baseline.

## Task 3 error analysis: outputs/error_analysis/{error_analysis.md,per_image.csv} (script evaluation/error_analysis.py)
L2: both right 39, Gemini only 35, Arm1 only 4, both wrong 22. L1: 75/23/2/0. "Why intersection is hard" section states hypotheses are untested (I did not view the images).

## Task 4 hero chart: outputs/figures/hero_avg_f1.{png,svg} (evaluation/make_hero_chart.py)
Needs matplotlib, which is in no repo venv; I installed it to scratchpad only (PYTHONPATH=.../scratchpad/mpl). Chart reads consolidated.json; rerun after the manager re-runs aggregate to add PaliGemma/other new rows. Footer states n=100.

## Other
Received a coordinator message meant for Worker A (Gemini key); ignored, not my lane.
