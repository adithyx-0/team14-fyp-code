# Consolidated evaluation — Malayalam held-out test set (n=100)

Macro-F1 with 95% percentile-bootstrap CIs (2000 resamples of the test set, seed 42). UNKNOWN predictions count as wrong. Avg-F1 = mean of level1 and level2 macro-F1. Seed ± is sample std (ddof=1). Bootstrap convention: a resample with no instance of a class scores that class F1=0 (affects ~1.7% of L1 resamples, 0.96^100, slightly widening the lower CI).

| System | Level-1 macro-F1 [95% CI] | Level-2 macro-F1 [95% CI] | Avg-F1 [95% CI] | UNKNOWN L1/L2 |
|---|---|---|---|---|
| Majority-class baseline (train mode: troll/person) | 0.4898 [0.479, 0.497] | 0.2338 [0.204, 0.260] | 0.3618 [0.346, 0.375] | 0% / 0% |
| Arm 1 — Swin-Tiny + MuRIL (classical) | 0.5076 [0.422, 0.611] | 0.3940 [0.299, 0.490] | 0.4508 [0.386, 0.514] | 0% / 0% |
| SYNAPSE replication — Qwen3-VL-2B LoRA (mlx) | 0.6564 [0.482, 0.870] | 0.3183 [0.243, 0.388] | 0.4874 [0.375, 0.602] | 0% / 0% |
| Gemini 3.5 Flash-Lite zero-shot | 0.8947 [0.656, 1.000] | 0.6521 [0.541, 0.754] | 0.7734 [0.645, 0.850] | 0% / 0% |
| arm2_vlm_benchmark/qwen3vl2b_cot | 0.4819 [0.468, 0.495] | 0.4289 [0.319, 0.528] | 0.4554 [0.399, 0.505] | 0% / 0% |
| arm2_vlm_benchmark/qwen3vl2b_zeroshot | 0.4872 [0.474, 0.497] | 0.4986 [0.384, 0.601] | 0.4929 [0.435, 0.544] | 0% / 0% |
| arm3_peft_finetune/r16_s42 | 0.6299 [0.479, 0.831] | 0.2338 [0.204, 0.260] | 0.4318 [0.348, 0.536] | 0% / 0% |
| arm3_peft_finetune/r4_os_s42 | 0.6923 [0.485, 1.000] | 0.5051 [0.392, 0.603] | 0.5987 [0.457, 0.737] | 0% / 0% |
| arm3_peft_finetune/r4_os_s43 | 0.4898 [0.479, 0.497] | 0.4454 [0.346, 0.552] | 0.4676 [0.417, 0.521] | 0% / 0% |
| arm3_peft_finetune/r4_os_s44 | 0.4898 [0.479, 0.497] | 0.4960 [0.368, 0.607] | 0.4929 [0.429, 0.548] | 0% / 0% |
| arm3_peft_finetune/r4_s42 | 0.5928 [0.471, 0.770] | 0.4272 [0.300, 0.540] | 0.5100 [0.401, 0.619] | 0% / 0% |
| arm3_peft_finetune/r4_s43 | 0.6299 [0.476, 0.856] | 0.3425 [0.267, 0.414] | 0.4862 [0.387, 0.603] | 0% / 0% |
| arm3_peft_finetune/r4_s44 | 0.4898 [0.479, 0.497] | 0.2772 [0.221, 0.339] | 0.3835 [0.356, 0.415] | 0% / 0% |
| arm3_peft_finetune/r8_s42 | 0.6094 [0.474, 0.795] | 0.3288 [0.252, 0.401] | 0.4691 [0.377, 0.565] | 0% / 0% |
| *SYNAPSE paper (reported, not reproduced)* | 0.9200 (reported) | 0.4256 (reported) | 0.6728 (reported) | n/a |

## Arm 3 seed summary (mean ± std over seeds)

Single seeds are not reliable at n=100; judge a config by its seed mean, not its best seed. Seeds were not selected on the test set.

| Config | Seeds | Level-1 macro-F1 | Level-2 macro-F1 | Avg-F1 |
|---|---|---|---|---|
| r16 | 1 | 0.630 (n=1) | 0.234 (n=1) | 0.432 (n=1) |
| r4 | 3 | 0.571 ± 0.073 | 0.349 ± 0.075 | 0.460 ± 0.067 |
| r4_os | 3 | 0.557 ± 0.117 | 0.482 ± 0.032 | 0.520 ± 0.070 |
| r8 | 1 | 0.609 (n=1) | 0.329 (n=1) | 0.469 (n=1) |

## Paired bootstrap vs Arm 1

Same resampled test sets for every system, so each difference is paired. Two-sided p = 2·min(P(Δ≤0), P(Δ≥0)); the CI is on Δ = system − Arm 1. With n=100, differences whose CI contains 0 are not evidence that either system is better.

| System | Δ Level-1 [95% CI] | p | Δ Level-2 [95% CI] | p |
|---|---|---|---|---|
| Majority-class baseline (train mode: troll/person) | -0.0178 [-0.130, +0.073] | 0.873 | -0.1602 [-0.263, -0.060] | 0.003 |
| SYNAPSE replication — Qwen3-VL-2B LoRA (mlx) | +0.1488 [-0.107, +0.428] | 0.467 | -0.0756 [-0.192, +0.038] | 0.187 |
| Gemini 3.5 Flash-Lite zero-shot | +0.3871 [+0.194, +0.542] | 0.000 | +0.2582 [+0.146, +0.369] | 0.000 |
| arm2_vlm_benchmark/qwen3vl2b_cot | -0.0257 [-0.137, +0.065] | 0.756 | +0.0349 [-0.106, +0.164] | 0.662 |
| arm2_vlm_benchmark/qwen3vl2b_zeroshot | -0.0204 [-0.131, +0.069] | 0.839 | +0.1046 [-0.028, +0.229] | 0.129 |
| arm3_peft_finetune/r16_s42 | +0.1223 [-0.110, +0.392] | 0.497 | -0.1602 [-0.263, -0.060] | 0.003 |
| arm3_peft_finetune/r4_os_s42 | +0.1847 [-0.104, +0.549] | 0.447 | +0.1111 [-0.016, +0.232] | 0.107 |
| arm3_peft_finetune/r4_os_s43 | -0.0178 [-0.130, +0.073] | 0.873 | +0.0515 [-0.079, +0.184] | 0.463 |
| arm3_peft_finetune/r4_os_s44 | -0.0178 [-0.130, +0.073] | 0.873 | +0.1020 [-0.044, +0.237] | 0.179 |
| arm3_peft_finetune/r4_s42 | +0.0852 [-0.115, +0.319] | 0.581 | +0.0332 [-0.113, +0.175] | 0.685 |
| arm3_peft_finetune/r4_s43 | +0.1223 [-0.111, +0.409] | 0.503 | -0.0515 [-0.168, +0.061] | 0.357 |
| arm3_peft_finetune/r4_s44 | -0.0178 [-0.130, +0.073] | 0.873 | -0.1167 [-0.220, -0.013] | 0.029 |
| arm3_peft_finetune/r8_s42 | +0.1018 [-0.114, +0.354] | 0.529 | -0.0651 [-0.175, +0.048] | 0.237 |

## Per-class F1

| System | troll | support | person | party | intersection |
|---|---|---|---|---|---|
| Majority-class baseline (train mode: troll/person) | 0.980 | 0.000 | 0.701 | 0.000 | 0.000 |
| Arm 1 — Swin-Tiny + MuRIL (classical) | 0.867 | 0.148 | 0.526 | 0.433 | 0.222 |
| SYNAPSE replication — Qwen3-VL-2B LoRA (mlx) | 0.979 | 0.333 | 0.699 | 0.256 | 0.000 |
| Gemini 3.5 Flash-Lite zero-shot | 0.989 | 0.800 | 0.842 | 0.714 | 0.400 |
| arm2_vlm_benchmark/qwen3vl2b_cot | 0.964 | 0.000 | 0.633 | 0.320 | 0.333 |
| arm2_vlm_benchmark/qwen3vl2b_zeroshot | 0.974 | 0.000 | 0.649 | 0.433 | 0.414 |
| arm3_peft_finetune/r16_s42 | 0.974 | 0.286 | 0.701 | 0.000 | 0.000 |
| arm3_peft_finetune/r4_os_s42 | 0.985 | 0.400 | 0.733 | 0.270 | 0.512 |
| arm3_peft_finetune/r4_os_s43 | 0.980 | 0.000 | 0.731 | 0.480 | 0.125 |
| arm3_peft_finetune/r4_os_s44 | 0.980 | 0.000 | 0.565 | 0.523 | 0.400 |
| arm3_peft_finetune/r4_s42 | 0.963 | 0.222 | 0.726 | 0.222 | 0.333 |
| arm3_peft_finetune/r4_s43 | 0.974 | 0.286 | 0.709 | 0.318 | 0.000 |
| arm3_peft_finetune/r4_s44 | 0.980 | 0.000 | 0.711 | 0.121 | 0.000 |
| arm3_peft_finetune/r8_s42 | 0.969 | 0.250 | 0.716 | 0.270 | 0.000 |

## Level-2 confusion matrices (rows = gold, columns = predicted)

**Majority-class baseline (train mode: troll/person)**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 54 | 0 | 0 | 0 |
| party | 31 | 0 | 0 | 0 |
| intersection | 15 | 0 | 0 | 0 |

intersection recall: 0%

**Arm 1 — Swin-Tiny + MuRIL (classical)**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 25 | 14 | 15 | 0 |
| party | 8 | 13 | 10 | 0 |
| intersection | 8 | 2 | 5 | 0 |

intersection recall: 33%

**SYNAPSE replication — Qwen3-VL-2B LoRA (mlx)**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 51 | 3 | 0 | 0 |
| party | 26 | 5 | 0 | 0 |
| intersection | 15 | 0 | 0 | 0 |

intersection recall: 0%

**Gemini 3.5 Flash-Lite zero-shot**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 48 | 3 | 3 | 0 |
| party | 5 | 20 | 6 | 0 |
| intersection | 7 | 2 | 6 | 0 |

intersection recall: 40%

**arm2_vlm_benchmark/qwen3vl2b_cot**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 38 | 9 | 7 | 0 |
| party | 20 | 8 | 3 | 0 |
| intersection | 8 | 2 | 5 | 0 |

intersection recall: 33%

**arm2_vlm_benchmark/qwen3vl2b_zeroshot**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 36 | 14 | 4 | 0 |
| party | 14 | 13 | 4 | 0 |
| intersection | 7 | 2 | 6 | 0 |

intersection recall: 40%

**arm3_peft_finetune/r16_s42**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 54 | 0 | 0 | 0 |
| party | 31 | 0 | 0 | 0 |
| intersection | 15 | 0 | 0 | 0 |

intersection recall: 0%

**arm3_peft_finetune/r4_os_s42**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 44 | 1 | 9 | 0 |
| party | 18 | 5 | 8 | 0 |
| intersection | 4 | 0 | 11 | 0 |

intersection recall: 73%

**arm3_peft_finetune/r4_os_s43**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 49 | 5 | 0 | 0 |
| party | 19 | 12 | 0 | 0 |
| intersection | 12 | 2 | 1 | 0 |

intersection recall: 7%

**arm3_peft_finetune/r4_os_s44**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 26 | 28 | 0 | 0 |
| party | 7 | 23 | 1 | 0 |
| intersection | 5 | 6 | 4 | 0 |

intersection recall: 27%

**arm3_peft_finetune/r4_s42**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 53 | 1 | 0 | 0 |
| party | 27 | 4 | 0 | 0 |
| intersection | 12 | 0 | 3 | 0 |

intersection recall: 20%

**arm3_peft_finetune/r4_s43**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 50 | 4 | 0 | 0 |
| party | 24 | 7 | 0 | 0 |
| intersection | 13 | 2 | 0 | 0 |

intersection recall: 0%

**arm3_peft_finetune/r4_s44**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 54 | 0 | 0 | 0 |
| party | 29 | 2 | 0 | 0 |
| intersection | 15 | 0 | 0 | 0 |

intersection recall: 0%

**arm3_peft_finetune/r8_s42**

| gold \ pred | person | party | intersection | UNKNOWN |
|---|---|---|---|---|
| person | 53 | 1 | 0 | 0 |
| party | 26 | 5 | 0 | 0 |
| intersection | 15 | 0 | 0 | 0 |

intersection recall: 0%

## Notes

- Level-1 is 95%+ troll in train, so accuracy is uninformative; macro-F1 weights the rare class equally (see README).
- The test set has only 4 `support` examples, so Level-1 macro-F1 rests on 4 images: one image moves it by tens of points, and Level-1 CIs are wide.
- There is no validation split (train 500 / test 100); no config was tuned on a held-out set, and a single best seed picked after seeing test scores is a lead, not a result.
- Level-1 and level-2 test-label distributions come from `test.csv`; every system's own gold columns were checked against it.
- Our macro-F1 averages over the gold classes only, so an UNKNOWN prediction is simply a miss. (`gemini_zeroshot.py`'s own printed F1 would also average over an 'UNKNOWN' class if any occurred; the numbers coincide when there are none.)
- Paper row is the figure reported in SYNAPSE@DravidianLangTech 2026; it was not re-run here.
