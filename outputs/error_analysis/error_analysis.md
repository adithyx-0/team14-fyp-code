# Error analysis: Gemini 3.5 Flash-Lite zero-shot vs Arm 1 (Malayalam test, n=100)

Source: `gemini_zeroshot_predictions.csv`, `arm1_classical/test_predictions.json`. Per-image table: `per_image.csv`.

## Level 1 agreement (correct/incorrect), n=100

| | Arm 1 correct | Arm 1 wrong |
|---|---|---|
| Gemini correct | 75 | 23 |
| Gemini wrong | 2 | 0 |

Gemini accuracy 98/100, Arm 1 accuracy 77/100. Accuracy is shown for the agreement table only; headline metric is macro-F1.

## Level 2 agreement (correct/incorrect), n=100

| | Arm 1 correct | Arm 1 wrong |
|---|---|---|
| Gemini correct | 39 | 35 |
| Gemini wrong | 4 | 22 |

Gemini accuracy 74/100, Arm 1 accuracy 43/100. Accuracy is shown for the agreement table only; headline metric is macro-F1.

## Level 2 accuracy per gold class

| gold | n | Gemini correct | Arm 1 correct |
|---|---|---|---|
| person | 54 | 48 | 25 |
| party | 31 | 20 | 13 |
| intersection | 15 | 6 | 5 |

### Gemini level-2 confusion (rows gold, cols predicted)

| gold \ pred | person | party | intersection |
|---|---|---|---|
| person | 48 | 3 | 3 |
| party | 5 | 20 | 6 |
| intersection | 7 | 2 | 6 |

### Arm 1 level-2 confusion (rows gold, cols predicted)

| gold \ pred | person | party | intersection |
|---|---|---|---|
| person | 25 | 14 | 15 |
| party | 8 | 13 | 10 |
| intersection | 8 | 2 | 5 |

## Example images (level 2; first 5 by test-file order in each direction; view `data/primary/malayalam/<image>`)

### Gemini right, Arm 1 wrong (35 total)

| image | gold | Gemini | Arm 1 |
|---|---|---|---|
| images/test/22.jpg | person | person | intersection |
| images/test/41.jpg | party | party | intersection |
| images/test/53.jpg | person | person | party |
| images/test/92.jpg | person | person | intersection |
| images/test/106.jpg | party | party | intersection |

### Arm 1 right, Gemini wrong (4 total)

| image | gold | Gemini | Arm 1 |
|---|---|---|---|
| images/test/15.jpg | party | intersection | party |
| images/test/65.jpg | intersection | person | intersection |
| images/test/202.jpg | party | person | party |
| images/test/231.jpg | party | person | party |

### Level 1: the 4 support images

| image | Gemini | Arm 1 |
|---|---|---|
| images/test/2.jpg | support | troll |
| images/test/53.jpg | support | support |
| images/test/92.jpg | support | support |
| images/test/328.jpg | support | troll |

## Why `intersection` is hard

- Only 15 of 100 test images are `intersection` (train: see class counts in DEVIATIONS/summary), so per-class F1 has very few examples.
- Gemini predictions on gold-intersection images: {'person': 7, 'party': 2, 'intersection': 6}; Arm 1: {'person': 8, 'intersection': 5, 'party': 2}.
- When Gemini says `intersection`, the gold labels are: {'party': 6, 'person': 3, 'intersection': 6} (precision 6/15).
- Observations from the table, not from inspecting every image: `intersection` is defined by a person AND their party being targeted together, which is a judgment about the relationship between two entities rather than a visual category; `person` vs `intersection` differs by whether party symbols/names are salient and Malayalam text that names the party is unreadable to a frozen image encoder and (per the pooled CV) the OCR text branch adds nothing. Annotator-boundary ambiguity between the three target classes is plausible but we did not measure inter-annotator agreement.
- Hypotheses above are untested; qualitative review of the listed image ids would confirm or refute them.
