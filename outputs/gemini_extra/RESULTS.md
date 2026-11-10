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
