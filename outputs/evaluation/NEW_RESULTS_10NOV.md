# New results, 10 Nov 2026 (Manager-assembled; consolidated.md/json untouched)

Same 100-image Malayalam test set, same bootstrap (2000 resamples, seed 42, percentile 95% CI) as `consolidated.md`. UNKNOWN counted as wrong. Numbers recomputed from raw prediction files.

## A. Malayalam test (n=100)

| System | L1 macro-F1 [CI] | L2 macro-F1 [CI] | Avg | UNK L1/L2 | dAvg vs Arm 1 [CI] | L2 person/party/intersection F1 |
|---|---|---|---|---|---|---|
| Arm 1 (reference) | 0.508 [0.422, 0.611] | 0.394 [0.299, 0.490] | 0.451 [0.386, 0.514] | 0% / 0% |  | 0.53 / 0.43 / 0.22 |
| Gemini 3.5 Flash-Lite zero-shot (2nd run, from zs600 test split) | 0.856 [0.630, 1.000] | 0.646 [0.538, 0.745] | 0.751 [0.632, 0.829] | 0% / 0% | +0.300 [+0.191, +0.387] | 0.82 / 0.69 / 0.42 |
| Gemini 3.5 Flash-Lite CoT | 0.942 [0.745, 1.000] | 0.657 [0.553, 0.751] | 0.799 [0.692, 0.861] | 0% / 0% | +0.349 [+0.243, +0.434] | 0.81 / 0.73 / 0.43 |
| Gemini 3.5 Flash-Lite few-shot (6 train exemplars) | 0.856 [0.645, 1.000] | 0.725 [0.623, 0.818] | 0.790 [0.669, 0.874] | 0% / 0% | +0.339 [+0.226, +0.441] | 0.82 / 0.73 / 0.62 |
| PaliGemma2-3B-mix-448 zero-shot (8-bit MLX) | 0.420 [0.225, 0.585] | 0.334 [0.257, 0.418] | 0.377 [0.267, 0.473] | 62% / 13% | -0.074 [-0.166, +0.010] | 0.67 / 0.06 / 0.27 |

Paired differences use the same resample indices for both systems (paired bootstrap). The test set has only 4 `support` items (96 troll), so L1 CIs are wide.

## B. Gemini extra runs: worker's RESULTS.md (600-image zero-shot, Tamil; verbatim)

# Gemini 3.5 Flash-Lite extra runs (Worker A)

Macro-F1, 95% percentile bootstrap (2000 resamples, seed 42). UNKNOWN counted as wrong.

| run / subset | n | L1 macro-F1 [CI] | L2 macro-F1 [CI] | UNKNOWN |
|---|---|---|---|---|
| cot | 100 | 0.942 [0.745, 1.000] | 0.657 [0.553, 0.751] | 0 |
| fewshot | 100 | 0.856 [0.645, 1.000] | 0.725 [0.623, 0.818] | 0 |
| tamil (L2 classes present: person/party) | 101 | 0.712 [0.482, 0.870] | 0.570 [0.442, 0.686] | 0 |
| tamil majority baseline (troll/person, mode of the Tamil test gold) | 101 | 0.490 [0.479, 0.498] | 0.448 [0.423, 0.471] | 0 |
| zs600 | 600 | 0.761 [0.688, 0.823] | 0.617 [0.574, 0.658] | 0 |
| zs600 [train-500] | 500 | 0.746 [0.662, 0.814] | 0.610 [0.560, 0.658] | 0 |
| zs600 [test-100] | 100 | 0.856 [0.630, 1.000] | 0.646 [0.538, 0.745] | 0 |


## C. Tamil transfer of Arm 1 (Worker B, verbatim JSON)

```json
{
 "n": 201,
 "dataset": "Tamil POLITICAL memes (not misogyny set)",
 "ocr": "tesseract tam+eng",
 "ocr_nonempty": 168,
 "level1": {
  "gold_classes": [
   "troll",
   "support"
  ],
  "macro_f1": 0.47429653679653677,
  "ci95": [
   0.4159030585052994,
   0.5388595979954646
  ],
  "majority_class": "troll",
  "majority_f1": 0.4654255319148936,
  "majority_ci95": [
   0.45081967213114754,
   0.4779220779220779
  ],
  "per_class_f1": {
   "troll": 0.8273809523809523,
   "support": 0.12121212121212122
  },
  "pred_counts": {
   "troll": 161,
   "support": 40
  },
  "gold_counts": {
   "troll": 175,
   "support": 26
  }
 },
 "level2": {
  "gold_classes": [
   "person",
   "party"
  ],
  "macro_f1": 0.4550620482179417,
  "ci95": [
   0.381781322280488,
   0.5265896176637714
  ],
  "majority_class": "person",
  "majority_f1": 0.44166666666666665,
  "majority_ci95": [
   0.4224137931034483,
   0.4582210242587601
  ],
  "per_class_f1": {
   "person": 0.6463878326996197,
   "party": 0.26373626373626374
  },
  "pred_counts": {
   "person": 104,
   "party": 49,
   "intersection": 48
  },
  "gold_counts": {
   "person": 159,
   "party": 42,
   "intersection": 0
  }
 }
}
```

