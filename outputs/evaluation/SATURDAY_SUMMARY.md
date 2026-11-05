# Team 14 — status summary (Saturday)

Task: Malayalam political memes, two labels. Level 1 = troll vs support; Level 2 = target person / party / intersection. All numbers: macro-F1 on the same 100-image held-out test set (96 troll / 4 support at L1; 54 / 31 / 15 at L2). Source: `outputs/evaluation/consolidated.md`. 95% bootstrap CIs are about ±0.1 at L2 and wider at L1.

## What works
- All three arms and the explainability pipeline are implemented and have run end to end. Because of no A100, Arms 2 and 3 use Qwen3-VL-2B (8-bit, MLX) instead of 7B models (`DEVIATIONS.md`).
- One evaluation script scores every system identically (bootstrap CIs, paired tests, per-class F1, confusion matrices).
- A web page (`website/`) runs Arm 1 with Grad-CAM on an uploaded meme.

## Honest numbers

| System | L1 macro-F1 | L2 macro-F1 | Avg |
|---|---|---|---|
| Majority-class baseline | 0.490 | 0.234 | 0.362 |
| Arm 1: Swin-Tiny + MuRIL | 0.508 | 0.394 | 0.451 |
| Arm 2: Qwen3-VL-2B zero-shot | 0.487 | 0.499 | 0.493 |
| Arm 2: Qwen3-VL-2B CoT | 0.482 | 0.429 | 0.455 |
| **Arm 2: Gemini 3.5 Flash-Lite zero-shot (API)** | **0.895** [0.656, 1.000] | **0.652** [0.541, 0.754] | **0.773** [0.645, 0.850] |
| Arm 3: LoRA rank 4, mean ± std over seeds 42/43/44 | 0.571 ± 0.073 | 0.349 ± 0.075 | 0.460 ± 0.067 |
| Arm 3: LoRA rank 4 + oversampling, seed 42 only | 0.692 | 0.505 | 0.599 |
| Arm 3: LoRA rank 4 + oversampling, seeds 43 / 44 | 0.490 / 0.490 | 0.445 / 0.496 | 0.468 / 0.493 |
| Arm 3: LoRA rank 4 + oversampling, mean ± std over 3 seeds | 0.557 ± 0.117 | 0.482 ± 0.032 | 0.520 ± 0.070 |

(std = sample std, n = 3 seeds.) Single runs: LoRA r8 seed 42 Avg 0.469, r16 seed 42 Avg 0.432 (r16 L2 equals the majority baseline).

**Grad-CAM faithfulness (Arm 1, n = 100, deletion/insertion vs random patch order, 5 random seeds):**
- Level 1: deletion AUC 0.588 vs 0.791 random (lower is better); insertion 0.864 vs 0.736 (higher is better).
- Level 2: deletion 0.354 vs 0.551; insertion 0.627 vs 0.511.
- Grad-CAM beats random on 82–93% of images, paired Wilcoxon p < 1e-11. Maps are coarse (7×7).

## What is weak
- **Level-1 is nearly unscorable.** Only 4 support images in the test set (23 of 500 in train). One image moves L1 macro-F1 by a large amount; CIs reach up to 1.00.
- **Seed variance is as large as the gaps between methods.** The same rank-4 setup spans Avg 0.38–0.51 across seeds.
- **No system is significantly better than Arm 1,** except against the majority baseline at L2. Plain LoRA often collapses to troll/person.
- **Gemini result needs care.** Gemini 3.5 Flash-Lite zero-shot is the best system by a wide margin and is significantly better than Arm 1 at both levels (paired bootstrap Δ L1 +0.387, Δ L2 +0.258, p<0.001, 0 UNKNOWN). But: (a) it is a Flash-Lite model, not the planned Gemini 2.5 Flash, which the API now returns 404 for new users (`DEVIATIONS.md`); (b) L1 still rests on 4 support images (CI 0.66-1.00); (c) it is a closed API model, so we cannot rule out that it has seen these public memes; (d) it reads the original images, the local Qwen arms read 512px resized copies. Its L2 per-class F1 for intersection is only 0.40.
- Cross-lingual transfer, Hateful Memes and the Tamil+Malayalam set are not run (datasets not obtained).
- **Oversampling gain is not established.** Seeds 43/44 finished: Avg 0.520 ± 0.070 vs plain r4 0.460 ± 0.067, a 0.06 gap smaller than the seed std. The 0.692 L1 came from seed 42 only; seeds 43/44 sit at the majority-baseline L1 (0.490). L2 is more encouraging (0.482 ± 0.032 vs 0.349 ± 0.075) but every oversampled seed's CI vs Arm 1 contains 0 (p = 0.11 / 0.46 / 0.18).
- r8 and r16 have one seed each, so they cannot be compared with r4.

## Gap to the paper (SYNAPSE, reported not reproduced)
- Paper: L1 0.9200, L2 0.4256, Avg 0.6728 (quoted from our replication notes; the paper itself is not in the folder).
- Our direct replication of its recipe: L1 0.656, L2 0.318.
- Our best single run among trained models: L1 0.692, L2 0.505, but one seed and CIs overlap the paper's L2 number.
- Gemini 3.5 Flash-Lite zero-shot: L1 0.895, L2 0.652, Avg 0.773, above the paper's reported Avg 0.673 on our split. This is only a comparison of numbers; the split match is unverified.
- We cannot say we match or beat the paper. Whether its split equals our 100 images is **unverified** (see `DEVIATIONS.md`, "Level-1 comparability").

## Pooled 5-fold x 10-repeat CV on all 600 images (Arm 1 heads, `cv_pooled/RESULTS.md`)
- Arm 1 beats the majority class at both levels (L2 p<0.001), which the 100-image test could not show for L1.
- The OCR text branch adds nothing measurable: image-only equals image+text (L1 0.566 vs 0.565, L2 0.405 vs 0.405, p~0.9).
- Re-run with text features standardised on each training fold: text-only improves (L1 0.468 -> 0.509, L2 0.285 -> 0.347), but fused is still no better than image-only (L1 -0.018, p=0.43; L2 -0.003, p=0.82). So the null result is not just an artefact of the near-constant frozen MuRIL [CLS] features, though a frozen [CLS] pathway is still a weak text encoder.
- Official-split seed spread of Arm 1 (10 seeds): L1 0.580 ± 0.059, L2 0.374 ± 0.054. The single published Arm 1 run sits inside it, so ~0.1 differences between single-seed systems on 100 images are noise.
