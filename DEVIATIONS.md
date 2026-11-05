# Hardware-constrained deviations from the original plan

The plan (`instructions.md`) assumed an A100. None is available, so each
affected item was replaced by the closest feasible experiment on a 16GB
M-series MacBook (plus the Gemini API). Nothing was silently dropped; every
substitution is listed here with the reason and what it still demonstrates.

| Planned | Done instead | Why this substitute | What it still shows |
|---|---|---|---|
| Arm 2: Qwen2.5-VL-7B, LLaVA-1.6-7B, PaliGemma-3B, Gemini 2.5 Flash, zero/few-shot/CoT | Qwen3-VL-2B-Instruct (8-bit MLX), zero-shot + CoT (`arm2_vlm_benchmark/local_vlm.py`); Gemini 2.5 Flash zero-shot script ready (needs API key) | 7B bf16 does not fit 16GB. Qwen3-VL-2B is the model the SYNAPSE paper itself fine-tunes, so Arm 2 (prompt-only) and Arm 3 (LoRA) are a controlled comparison on one backbone. | Value of prompting vs fine-tuning, same model, same 100 test images. Few-shot omitted: multi-image prompts exceed the 2B model's useful context on this hardware. |
| Arm 3: QLoRA (4-bit) on three 7B VLMs, 30-40 run ablation grid | LoRA on Qwen3-VL-2B (8-bit base), 6-run grid: rank {4,8,16}, seeds {42,43,44}, +minority oversampling (`arm3_peft_finetune/run_ablation.py`) | Same recipe as the paper (alpha=2*rank, dropout 0.15, q/v_proj, lr 5e-5, wd 0.1, eff. batch 8, 500 steps). One run takes ~10 min here, making a real grid feasible. | The plan's ablation axes (rank, seed, imbalance handling) at 2B scale. Seed repeats give the noise floor needed to interpret a 100-example test set. |
| Explainability on VLM vision towers (7B) | Grad-CAM + deletion/insertion faithfulness on Arm 1 (Swin-Tiny); shared schema module | Grad-CAM on a 7B VLM needs gradient access through a bf16 7B model. | One shared `ExplanationRecord` schema (Risk 1) so VLM rationales/tokens plug in later. |
| Demo with all arms live | Gradio demo: Arm 1 live + Grad-CAM; other arms show precomputed test-set predictions only | Honest labelling instead of simulated outputs. | Visible model dropdown (Risk 5). |
| Hateful Memes reference pass; Tamil+Malayalam misogyny secondary set | Not run | Datasets never acquired (gated / registration lag), unrelated to compute. | Listed as future work. |
| Cross-lingual transfer | Not run (unless noted in results) | Needs the secondary dataset for checkpoint transfer. | — |

## Comparability controls (why the numbers can be set side by side)
- All systems are scored on the **same 100-example Malayalam test set**
  (ids and labels verified identical between `data/primary/malayalam/test.csv`
  and the replication's `test.csv`: 100/100 match).
- Metric: macro-F1 per level (class-imbalanced data; accuracy would reward
  predicting the majority class), with 95% bootstrap CIs and paired
  bootstrap tests (`evaluation/`), because n=100 makes point estimates noisy.
- Label vocabulary is normalised once in `evaluation/aggregate.py`
  (SUPPORT/CRITICISE -> support/troll; PERSON/PARTY/BOTH -> person/party/intersection).
  Unparseable outputs count as wrong and the UNKNOWN rate is reported.

## Imbalance handling per arm (Risk 3 log)
- Arm 1: class-weighted cross-entropy.
- Arm 2: none (no training); prompts are label-neutral.
- Arm 3: baseline none; one ablation arm oversamples party x2 / intersection x4
  in the training set only (test set untouched).

## Level-1 comparability with the SYNAPSE paper (0.92)

Source limits: the SYNAPSE paper PDF is **not** in the project folder. What
follows comes from our own replication notes (`../test fyp/Notes0.pdf`), which
quote the paper, plus our local data files. Anything not in those is marked
UNVERIFIED. Nothing here was checked against the paper itself.

| Item | Finding | Status |
|---|---|---|
| Paper's Malayalam scores | L1-F1 0.9200, L2-F1 0.4256, Avg-F1 0.6728, shared-task rank 5 (Notes0 results table) | Quoted via our notes, not read in the paper |
| Metric | Macro-averaged P/R/F1 per level; ranking metric Avg-F1 = (L1+L2)/2. 0.6728 = (0.9200 + 0.4256)/2 exactly, consistent with macro-F1 | Notes0 states it; arithmetic check ours |
| Paper's test split size / class counts | Not stated in any file we have | **UNVERIFIED** |
| Is our 100-image test set the paper's test set? | Our replication used the real DravidianLangTech 2026 held-out Malayalam test set (Notes0), and our `data/primary/malayalam/test.csv` matches it 100/100. Whether the paper's reported 0.92 was scored on this same split is not confirmed | **UNVERIFIED** |
| Our test set composition | 100 images: 96 troll, 4 support. Train: 477 troll, 23 support (counted from `test.csv` / `train.csv`) | Verified locally |
| Same hyperparameters/prompt as paper | Replication used the paper's settings (lr 5e-5, wd 0.1, eff. batch 8, 500 steps, structured INTENT/TARGET output) per Notes0, but at 8-bit on MLX rather than bf16 on GPU | Per Notes0; deviation documented above |

What this implies (reasoning, not a verified finding):
- With 4 support images, Level-1 macro-F1 is dominated by how many of those 4
  are found. Our replication of the paper's recipe scored L1 0.6564 with 96%
  accuracy and support recall 1/4 (Notes0). Our best run is 0.69 and the
  bootstrap CI reaches 1.00 (`outputs/evaluation/consolidated.md`).
- A score of 0.92 would need nearly all 4 support images correct, if the paper
  used this same split. If its split differs, the numbers are not comparable.
- Treat the paper's L1 as **reported, not reproduced and not shown to be
  comparable**. The Level-2 comparison (paper 0.4256) carries the same
  caveat about the split, though it has more support per class.
- To close this: obtain the paper (or contact the authors) for the split and
  class counts. Until then do not claim we are above or below the paper.
