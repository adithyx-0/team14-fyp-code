# Manager report, 10 Nov 2026

Both workers have finished. Worker B completed all four tasks. Worker A completed 3 of 5 and is blocked on the Gemini free-tier daily quota (500 requests/day/model).
The key blocker (missing .env.gemini) was resolved at about 12:55. The key file is present and gitignored.

## Per-task status

| Task | Status | Headline (Malayalam test n=100 unless stated; 95% bootstrap CI, UNKNOWN = wrong) |
|---|---|---|
| A1 Gemini zero-shot on all 600 images | DONE | n=600: L1 0.761 [0.688, 0.823], L2 0.617 [0.574, 0.658]. Train-500: 0.746 / 0.610. Test-100 (2nd run): 0.856 [0.630, 1.000] / 0.646 [0.538, 0.745]. 0 UNKNOWN, 0 API errors. |
| A2 Gemini CoT | DONE | L1 0.942 [0.745, 1.000], L2 0.657 [0.553, 0.751], Avg 0.799. 0 UNKNOWN. |
| A3 Gemini few-shot (6 train exemplars) | DONE | L1 0.856 [0.645, 1.000], L2 0.725 [0.623, 0.818], Avg 0.790. 0 UNKNOWN. |
| A4 Gemini Tamil zero-shot | PARTIAL, 101 of 201 | n=101 (the first 101 rows of test.csv): L1 0.712 [0.482, 0.870], L2 0.570 [0.442, 0.686]. Only 4 `support` items, so L1 is not meaningful. |
| A5 Gemini rationales | NOT RUN | Blocked by the quota. |
| B1 PaliGemma2-3B-mix-448 zero-shot (8-bit MLX, n=100) | DONE | L1 0.420 [0.225, 0.585], L2 0.334 [0.257, 0.418], Avg 0.377. UNKNOWN: 62% at L1, 13% at L2. Below the majority baseline. |
| B2 Arm 1 Malayalam to Tamil transfer (n=201) | DONE | L1 0.474 [0.416, 0.539] vs majority 0.465. L2 0.455 [0.382, 0.527] vs majority 0.442. This is no transfer beyond the majority baseline. |
| B3 Error analysis, Gemini zs vs Arm 1 | DONE | At L2, both are right on 39, Gemini only on 35, Arm 1 only on 4, and both are wrong on 22. At L1 the counts are 75 / 23 / 2 / 0. |
| B4 Hero chart | DONE, but stale | outputs/figures/hero_avg_f1.{png,svg}. It reads consolidated.json, which I did NOT change, so it lacks the new systems. |

### Integration
`outputs/evaluation/consolidated.{md,json}` is **untouched**. A backup is in `outputs/_coordination/backup/`.
I chose the low-risk route: `evaluation/new_results_10nov.py` (reuses the aggregate.py functions) writes `outputs/evaluation/NEW_RESULTS_10NOV.md`.

That file has paired-bootstrap differences against Arm 1 on the same 100 images:

| System | Avg macro-F1 (test) | Δ vs Arm 1 (Avg), 95% CI |
|---|---|---|
| Arm 1 | 0.451 | reference |
| Gemini zero-shot (2nd run) | 0.751 | +0.300 [+0.191, +0.387] |
| Gemini CoT | 0.799 | +0.349 [+0.243, +0.434] |
| Gemini few-shot | 0.790 | +0.339 [+0.226, +0.441] |
| PaliGemma | 0.377 | -0.074 [-0.166, +0.010] |

## Independent audit (sklearn recomputation from raw prediction files)
- **Recomputed numbers.** Every headline number below matches the workers' reports to the 3rd decimal:
  - zs600 (all, train, test)
  - CoT
  - few-shot
  - Tamil Gemini
  - PaliGemma
  - Arm 1 Tamil
- **Raw output vs parsed label.** In every Gemini file, the parsed label matches the raw JSON output (0 mismatches).
- **Few-shot exemplars.** There are 6, one per (L1, L2) combination, seed 42, all from train: images/train/{287,243,596,166,97,36}.jpg. The script asserts that none are test ids, and the prompt contains no test ids. Pass.
- **UNKNOWN rates.** The files state them: 0 for Gemini, and 62% / 13% for PaliGemma.
- **Label mappings.**
  - Gemini maps SUPPORT/CRITICISE to support/troll and PERSON/PARTY/BOTH to person/party/intersection.
  - Tamil has NO intersection class. Predictions of "intersection" count as errors, and L2 macro-F1 is taken over person and party only. Both workers did this.
- **Key leak.** `grep AIza` and a fixed-string grep for the key's contents, over outputs/, docs, scripts and logs, found nothing. .env.gemini is gitignored.
- **Lanes and existing files.** `git diff --stat` shows only the .gitignore change (the 3-line .env.gemini entry); no existing result files were modified. No commits, pushes or adds were made.
- **Hand-edited numbers.** None found.

## Problems and how they were handled
1. **Missing key.** The key was added at about 12:55 and Worker A started after that.
2. **429 with 4 parallel jobs.** Worker A cut to one sequential chain. It stopped writing API failures as UNKNOWN and logged them as API errors with the row left unfilled. Earlier quota-UNKNOWN rows were stripped before the rerun. The final files have 0 error rows.
3. **Daily quota exhausted at about 14:40.** I probed the API directly and got 429: limit 500/day for gemini-3.5-flash-lite on the free tier. About 100 requests were wasted on retries and duplicate parallel work. Tamil is therefore partial and rationales were not run. The quota resets about 13:30 IST on 11 Nov, or the owner can enable billing on the key. To resume: `.venv/bin/python -m arm2_vlm_benchmark.gemini_extra tamil`, then `... rationale`, then `.venv/bin/python -m arm2_vlm_benchmark.gemini_eval`. The runs are resumable. The Thu 12 Nov report deadline probably cannot wait for that.
4. **Worker communication.** I could not address the workers directly (no ListAgents, and the names did not resolve). All instructions went via the coordinator.
5. **PaliGemma needed prompt changes.** The Qwen prompt returned "Sorry, as a base VLM...". The v3 prompt asks two separate questions per image, with INTENT worded "praise or mock". This is a different protocol from the other systems, so it is not strictly comparable.
6. **matplotlib.** matplotlib is installed in no repo venv. Worker B used a scratchpad install. The hero chart needs a rerun after consolidation.

## Caveats (must go into the report)
- **Test set size.** The test set is 100 images with only 4 `support` memes. All L1 CIs are wide, and L1 differences between the Gemini variants are meaningless. The Gemini variants also differ little from each other at the Avg level: CoT 0.799, few-shot 0.790 and zero-shot 0.751 overlap heavily. I did not run paired tests among them, so do not claim CoT or few-shot helps. The L2 intersection F1 is higher with few-shot (0.62 vs 0.42 zero-shot), which is suggestive only.
- **Non-determinism.** Gemini zero-shot on test gave L1 0.895 / L2 0.652 in the earlier run and 0.856 / 0.646 now, at temperature 0. Report it as run-to-run variation. The earlier published row is a different run from the zs600 test split.
- **Contamination.** Gemini's pretraining may include the memes. Train-500 and test-100 scores are within each other's CIs (L1 0.746 vs 0.856, L2 0.610 vs 0.646). This is no evidence of a test-specific advantage, but it does not rule out contamination of both.
- **Model.** The model is Flash-Lite (Gemini 3.5), not the Gemini 2.5 Flash named in the proposal. The proposal's Qwen2.5-VL-7B and LLaVA-7B were not run. PaliGemma is the 2-mix 448 variant (a mix-trained checkpoint), 8-bit.
- **Tamil.** These are political memes (troll/support, person/party), NOT misogyny, and not the proposal's dataset. There is no intersection class. Arm 1's Tamil OCR uses tam+eng Tesseract (168 of 201 non-empty). The Gemini Tamil subset is 101 images, and Arm 1 on the same 101 gets L1 0.433 / L2 0.470 (my calculation). The Tamil majority baselines are computed from the Tamil gold labels themselves. The Tamil result is NOT comparable to the Malayalam table.
- **Image inputs differ.** Arm 1 uses images plus Tesseract OCR text. Gemini gets the image files from data/primary/malayalam directly (no OCR stage, I did not verify their resolution). PaliGemma gets 512px images. The inputs are not identical across systems, so this is not an equal-footing comparison.
- **Arm 1 pooled CV.** Arm 1 pooled 600-image CV is L1 0.565 / L2 0.405 (Worker A's note). Gemini on the same 600 is L1 0.761 / L2 0.617. The gap is similar to the test-100 gap.
- **Error analysis.** The "why intersection is hard" section is explicitly untested hypotheses (images were not viewed).

## Files created
- arm2_vlm_benchmark/{gemini_extra.py, gemini_eval.py, paligemma_zeroshot.py}
- cross_lingual/ and evaluation/{error_analysis.py, make_hero_chart.py, new_results_10nov.py}
- outputs/gemini_extra/ (zs600, cot, fewshot, tamil jsonl + *_predictions.csv, fewshot_exemplars.json, RESULTS.md, logs)
- outputs/paligemma/ (paligemma2_3b_zeroshot.json plus smoke files)
- outputs/cross_lingual/ (tamil_transfer_results.json, tamil_test_predictions.json, tamil_ocr_cache.json)
- outputs/error_analysis/ (error_analysis.md, per_image.csv), outputs/figures/hero_avg_f1.{png,svg}
- outputs/evaluation/NEW_RESULTS_10NOV.md
- outputs/_coordination/ (workerA.md, workerB.md, backup/, this report)
- .gitignore: +3 lines (.env.gemini). Nothing is committed.

## What the owner should do next
1. **Decide on the key.** Either enable billing on the key (then run the Tamil and rationale commands above) or leave Tamil partial and rationales out. Do not commit .env.gemini.
2. **Report abstract and main text.** These still say Gemini was run zero-shot only on the 100 test images, and that "chain-of-thought" and "few-shot" are future work for the VLM.
   - Update to: Flash-Lite zero-shot 0.751 Avg (second run; the earlier run was 0.773), CoT 0.799, few-shot 0.790, all with n=100 and wide CIs.
   - State that prompting variants do not differ significantly.
   - The line "Natural-language rationales, cross-lingual transfer and a 7B-scale study are left to Phase II" is partly outdated. Cross-lingual Arm 1 to Tamil is done and gave a null result. Rationales are not done. 7B was not done.
3. **Add a 600-image row and a pooled comparison.** Gemini zero-shot on 600 gives L1 0.761 / L2 0.617, against Arm 1 pooled CV 0.565 / 0.405.
4. **Add PaliGemma.** Add the row to the VLM benchmark table. PaliGemma 0.377 Avg, 62% UNKNOWN at L1, with the prompt deviation footnoted.
5. **Add a Tamil paragraph.** Arm 1 zero-shot transfer is at the majority baseline (L1 0.474 vs 0.465, L2 0.455 vs 0.442). The data are political memes, not misogyny. Gemini on a 101-image subset: L1 0.712 and L2 0.570, with only 4 `support` items. Do not claim cross-lingual success.
6. **Fix the "4 support memes" wording consistency.** The test split has 4 `support` and 96 `troll`. (One worker note said ~7, which is wrong.)
7. **Regenerate the hero chart.** After you decide to fold the new rows into consolidated.md (extend FIXED_SYSTEMS in evaluation/aggregate.py, or just cite NEW_RESULTS_10NOV.md), rerun evaluation/make_hero_chart.py. matplotlib needs installing first.
8. **Update the action-plan table.** Item 2 of TEAM14_PHASE1_ACTION_PLAN.md is no longer ~20%, because Gemini CoT, few-shot and PaliGemma are now done (7B models are still missing). Item 6 is partially done as political-meme Tamil transfer, not the misogyny dataset.
9. **Do not edit the diary or .tex until you have read this.** Commits are yours to make.
