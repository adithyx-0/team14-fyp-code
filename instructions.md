# Team 14 FYP — Main Project Build Plan

**Project title:** LLM-Augmented Multimodal Meme Understanding: Stance Classification,
Explainability, and Cross-Lingual Analysis for Low-Resource Indian Languages

**This folder (`code/`)** is where the actual main-project implementation lives — separate
from `../test fyp/`, which is a *finished, standalone* SYNAPSE-paper replication used only
to produce a baseline comparison number. Do not redo that work here; reuse its outputs and
its lessons (see "What already exists" below).

Read this whole file before writing code. It is the single source of truth for what this
project is, what architecture it must follow, what's already done, what's not started, and
in what order to build it.

---

## 1. What this project actually is

This is **not** a single model — it's a comparative study across three parallel "arms"
that all attack the same task (two-level meme stance/target classification) with different
approaches, plus a shared explainability layer, cross-lingual transfer, and a demo app.
The novelty the paper/report claims is the *systematic multi-model comparison itself*, so
the architecture must make that comparison structurally visible, not bury it inside one
pipeline stage.

Full design rationale, the three architecture candidates considered, and the decision to
go with **Candidate B (parallel arms)** are in
`../Team14_System_Design_Review.md` — read that document for the *why*; this file is the
*what to build, in what order*.

### Non-negotiable functional units (all 9 must exist by the end)

1. **Data & augmentation layer** — primary, secondary, reference datasets; one frozen
   5-fold stratified split; three augmentation conditions (none / manual / LLM-synthetic)
   decided once, upstream of every arm.
2. **Arm 1 — Classical baseline**: frozen Swin-Tiny + BERT, residual fusion, 5-fold CV.
   This is the anchor every other result is measured against.
3. **Arm 2 — VLM benchmarking**: Qwen2.5-VL-7B, LLaVA-1.6-7B, PaliGemma-3B, Gemini 2.5
   Flash (API) — each under zero-shot, few-shot, and chain-of-thought prompting, both
   label levels.
4. **Arm 3 — PEFT fine-tuning**: LoRA/QLoRA (4-bit) on the three open-weight VLMs only
   (Gemini is API-only, can't be fine-tuned — see its own track below). Ablation on LoRA
   rank, adapter layer count, multi-seed variance.
5. **Synthetic data augmentation** — LLM-generated minority-class captioning pipeline,
   benchmarked against no-augmentation and manual-augmentation baselines.
6. **Explainability module** — Grad-CAM heatmaps + natural-language rationale generator,
   with an explicit faithfulness check between the two. **One shared module all arms call
   into** — not reimplemented per arm (see Risk 1 below).
7. **Cross-lingual transfer** — runs only on the *secondary* dataset (misogyny memes,
   Tamil+Malayalam); the primary dataset has no Tamil half. Two genuinely different
   sub-flows: checkpoint transfer (Arms 1/3) and prompt-language transfer (Arm 2) — do
   not treat as one uniform mechanism.
8. **Evaluation & comparison** — macro-F1 at both label levels, vs. the SYNAPSE baseline
   (0.4256, already reproduced — see below), vs. the Hateful Memes reference point, plus
   ablation and faithfulness results.
9. **Demo app** — Gradio/Streamlit. Accepts a meme image + optional Malayalam/Tamil
   caption, returns predicted label, confidence, Grad-CAM overlay, salient token
   highlights. **Model choice is a visible dropdown, not an automatic "best" pick** —
   see Risk 5.

### High-level data flow

```
Data & augmentation layer
        │
        ├──► Arm 1 (classical baseline)   ─┐
        ├──► Arm 2 (VLM benchmarking)      ─┼─► shared Explainability & Faithfulness layer
        └──► Arm 3 (PEFT fine-tuning)      ─┘         │
                                                        ▼
                                    Cross-lingual transfer & Evaluation
                                                        │
                                                        ▼
                                                    Demo app
```

---

## 2. What already exists — reuse, don't redo

`../test fyp/` is a **complete, real-data replication** of the SYNAPSE paper this project
is compared against. It is functionally a hardware-adapted proof of concept of what Arm 3
needs to do. Before writing any Arm 3 code here, read `../test fyp/README.md` and
`../test fyp/Notes0.html` in full.

**Directly reusable:**
- **The baseline number**: SYNAPSE Malayalam Level-2 macro-F1 = 0.4256 (Avg-F1 0.6728) is
  the paper's number the FYP objectives document names as the comparison target. Our own
  replication reproduced 0.3183 (Avg-F1 0.4874) on real data — cite both when reporting
  Arm 3 results, not just the paper's number.
- **The primary dataset, already downloaded, cleaned, and verified**:
  `../test fyp/data/{tamil,malayalam}/{train,test}.csv` and the resized 512px image sets.
  This is the *same* primary dataset Arm 1/2/3 need here — do not re-download or re-clean
  it; point this project's data layer at it (or copy it in, keeping the same schema:
  `image,level1,level2`).
- **Label-normalization lessons** (`Notes0.html` section 2): raw label text varies in
  casing/spacing across files; Codabench's "test data with labels" download was actually
  unlabeled (verify contents, never trust filenames); image filenames sometimes need to be
  re-derived from numeric IDs. Expect the *secondary* (misogyny) and *reference* (Hateful
  Memes) datasets to have their own version of this problem — apply the same discipline:
  exhaustively enumerate every raw label string, map explicitly, raise on unrecognized
  values, assert row counts match known image counts.
- **The 512px resize gotcha**: default image processors for these model families upsample
  small images rather than capping them — verify empirically per model, don't assume a
  resize step is redundant.
- **LoRA/QLoRA pipeline shape** (`scripts/{prepare_dataset,train_lora,infer,evaluate}.py`
  in `test fyp/`) — proven end-to-end structure (prepare → train → infer → evaluate) to
  mirror here, adapted to the paper's actual `transformers`+`peft`+CUDA stack now that
  A100 access exists, rather than `mlx-vlm`.

**Not reusable as-is:** the replication ran a 2B model at 8-bit on Apple Silicon via
`mlx-vlm`, a deliberate hardware compromise. This project's Arm 3 targets 7B-class models
on CUDA (A100) in bf16 — closer to the paper's own stack. Running Arm 3 in bf16 on real
CUDA hardware also gives the chance to test something the replication explicitly flagged
as unverified: whether quantization + mlx-vlm/Metal (vs. bf16 + transformers/CUDA) explains
part of the gap to the paper's numbers.

---

## 3. Datasets — status and what's needed

| Dataset | Role | Status |
|---|---|---|
| Malayalam political memes (500 train / 100 test, 2-level labels) | Primary — feeds Arms 1–3 | **Have it** — `../test fyp/data/malayalam/` |
| Tamil political memes (803 train / 201 test) | Used in the SYNAPSE replication; primary dataset proper is Malayalam-only per the FYP objectives doc — confirm with the team whether Tamil political memes are in scope here or were only needed for the replication | Have it, scope unclear — **confirm before building on it** |
| Tamil + Malayalam misogyny memes | Secondary — feeds cross-lingual transfer only | **Not acquired** — find source, register/request access, budget lag time for approval |
| Hateful Memes (English) | Reference — feeds Arm 2's zero-shot pass only, as a contextualization point, not a full benchmark | **Not acquired** — gated dataset, requires registration/license agreement (Meta/DrivenData) — start this early, approval can be slow |

Decide **now**, before Arm 2 work starts: will the Hateful Memes zero-shot pass run on the
full official test set, or a subsample? The full set is large enough to meaningfully change
Arm 2's total inference time (see Section 5) — a subsample is defensible since its stated
role is "contextualization point," not a full benchmark.

---

## 4. Compute resources

| Resource | Spec | Best use |
|---|---|---|
| College A100 (confirm 40GB vs 80GB) + 96GB RAM + fast SSD | Primary compute | Arm 2 (7B VLM inference), Arm 3 (QLoRA fine-tuning + ablation grid), explainability on 7B models |
| M5 MacBook (16GB unified memory, `mlx-vlm` already set up) | Secondary / immediate | Arm 1 (classical baseline — small enough to fully train locally), PaliGemma-3B (smallest open VLM, likely runs locally the way Qwen3-VL-2B did in the replication), Gemini API calls (no local compute needed), demo app scaffolding, eval/aggregation scripts, data layer engineering |
| Vast.ai (cloud rental, cheapest, interruptible) | ~$0.12–0.40/hr A100/4090 | Overflow capacity for the Arm 3 ablation grid if the college A100 is booked out — fine for short runs you can requeue if interrupted |
| RunPod Secure Cloud / Lambda Labs (cloud rental, reliable) | ~$1.39–1.99/hr A100 | Only for deadline-critical runs where an interruption would be costly — not the default |

**Before relying on the college A100:** confirm (1) 40GB vs 80GB, (2) whether it's a shared
booking system or dedicated access, (3) any per-job time limits. Run a short timed pilot
(a few training steps, same approach as the replication's Step 3 pilot) the first time you
get access, to convert the estimates in Section 5 into real numbers before committing to a
full run.

---

## 5. Timeline reality check

The design review (dated 1 Aug 2026) targeted Mid Review on **25 Aug 2026**, described then
as "~3.5 weeks out." **Confirm this date is still current before planning against it** — if
today is materially later than 1 Aug, the runway is shorter than the design doc assumed, and
the scope in Section 6 needs to shrink accordingly (recompute how many days remain and match
against the phase estimates below).

**Rough compute-time estimates** (raw GPU time, not counting debugging/integration):
- Arm 1 (5-fold CV): 1–2 hrs
- Arm 2 on primary dataset (4 models × 3 prompt strategies × 100 test images): 1–2 hrs
- Arm 2 reference pass (Hateful Memes, zero-shot only): 2–8 hrs depending on subsample size
- Arm 3 ablation grid (~30–40 runs across 3 models × rank × layer-count × seed): ~10–15 hrs
- Explainability (Grad-CAM + rationale, inference-time): 2–4 hrs
- Cross-lingual transfer: 3–6 hrs

Raw compute totals to roughly 1.5–2 days of GPU time. **Total wall-clock time is much
longer** — most of the real cost is debugging output parsing per model, getting Grad-CAM
hooks working across four different architectures, data-cleaning gotchas (expect the
secondary/reference datasets to have their own version of the label-normalization problems
hit in the replication), and integration — realistically 3–4+ weeks run serially, or
roughly 1.5–2 weeks if the three arms genuinely run in parallel across the team.

**If the runway is short (≤2 weeks to Mid Review), scope down to this MVP rather than
attempting the full design:**
- Arm 1 — fully done (cheap, low risk, do first).
- Arm 2 — primary dataset only; skip or heavily subsample the Hateful Memes reference pass.
- Arm 3 — one or two LoRA ranks per model, not the full ablation grid; enough to show
  fine-tuning works and to compare against Arm 2, not a full ablation study.
- Explainability — partial/stretch; even Grad-CAM on Arm 1 alone is a defensible Mid
  Review checkpoint.
- Cross-lingual transfer and the polished demo — defer to Phase II, post–Mid Review.

---

## 6. Build order

Work through phases in this order. Phases 1–2 can start immediately on the MacBook without
waiting on A100 access; do not block on compute access before starting them.

### Phase 0 — Setup (do first, no compute needed)
- [ ] Confirm the Mid Review date and how much runway actually remains.
- [ ] Confirm A100 access model (booking system? dedicated? time limits?).
- [ ] Start Hateful Memes and secondary-dataset access requests now — approval lag is the
      one thing that can't be sped up by working harder later.
- [ ] Set up this repo's structure (see Section 7).
- [ ] Copy or symlink the primary dataset in from `../test fyp/data/malayalam/` (and
      `tamil/` if in scope — see Section 3) rather than re-downloading.

### Phase 1 — Data & augmentation layer (frozen before any arm starts)
- [ ] Normalize and verify the secondary and reference datasets using the same discipline
      as the replication: enumerate every raw label string, map explicitly, raise on
      unrecognized values, verify every image file resolves on disk, assert row counts.
- [ ] Build the canonical 5-fold stratified split — **once**, shared by all three arms.
- [ ] Implement and freeze the three augmentation conditions (none / manual /
      LLM-synthetic) — decided here, not independently per arm (this was Risk 3 in the
      design review; if each arm decides imbalance-handling independently, the three arms
      stop being comparable).
- [ ] Log which imbalance-handling technique each arm ends up using, so evaluation can
      report it as a factor rather than hide it.

### Phase 2 — Arm 1: Classical baseline (MacBook-feasible, start immediately)
- [ ] Swin-Tiny + BERT, residual fusion, 5-fold CV on the primary dataset.
- [ ] This is the anchor number every other arm is compared against — get it right and
      get it first.

### Phase 3 — Arm 2: VLM benchmarking
- [ ] Gemini 2.5 Flash zero/few-shot/CoT pipeline — buildable on the MacBook now (API-only,
      no local compute).
- [ ] PaliGemma-3B — likely runs locally via `mlx-vlm` the same way the replication's
      Qwen3-VL-2B did; try this on the MacBook before assuming A100 is required.
- [ ] Qwen2.5-VL-7B, LLaVA-1.6-7B — needs A100. Zero-shot/few-shot/CoT, both label levels.
- [ ] Zero-shot pass on the Hateful Memes reference set, per the Phase 0/Section 3 scoping
      decision on subsample size.

### Phase 4 — Arm 3: PEFT fine-tuning
- [ ] Port the replication's prepare → train → infer → evaluate pipeline shape to
      `transformers` + `peft` + `bitsandbytes` (QLoRA 4-bit) on CUDA, matching the paper's
      exact stack now that real GPU hardware is available.
- [ ] Same LoRA config as the paper/replication as the starting point: rank=4, alpha=8,
      dropout=0.15, applied to q_proj/v_proj only — then run the ablation (rank, adapter
      layer count, seed) on top of that baseline, scoped per Section 5's timeline reality.
- [ ] Gemini gets a **parallel adaptation track**, not fine-tuning: heavier few-shot/CoT
      prompt optimization, since it's API-only and can't be fine-tuned (Risk 2 from the
      design review — don't silently drop it from the comparison).

### Phase 5 — Explainability & faithfulness (shared module, build once)
- [ ] One module all three arms call into — not reimplemented per arm (Risk 1). Type-aware
      output schema: `image_regions` for anything with a vision backbone, `salient_tokens`
      for anything with a rationale generator, both optional, both feeding one faithfulness
      scorer.
- [ ] Grad-CAM for Arm 1 and any VLM vision tower where gradients are exposed.
- [ ] NL rationale generation for Arm 2/3.
- [ ] Faithfulness scoring between the two.

### Phase 6 — Cross-lingual transfer
- [ ] Split explicitly into two sub-flows (Risk 4) — do not implement as one uniform
      mechanism:
  - Checkpoint transfer (Arms 1 and 3) on the secondary dataset.
  - Prompt-language transfer (Arm 2 — frozen models, just swap prompt language).

### Phase 7 — Evaluation & comparison
- [ ] Aggregate macro-F1 (both levels) across all arms.
- [ ] Compare against SYNAPSE (0.4256 paper / 0.3183 our replication) and the Hateful Memes
      reference point.
- [ ] Report ablation and faithfulness results.

### Phase 8 — Demo app
- [ ] Gradio/Streamlit, buildable and testable with stub predictions well before final
      models are ready (can start scaffolding on the MacBook in parallel with earlier
      phases).
- [ ] Accepts Malayalam/Tamil meme + optional caption.
- [ ] **Model choice is a visible dropdown** (Risk 5) — do not implement automatic "best
      model" selection; with 3 arms × 2 label levels × 2 languages, "best" is ambiguous and
      picking one silently pre-empts the comparison the project is supposed to make.
- [ ] Returns predicted label, confidence, Grad-CAM overlay, salient token highlights.

---

## 7. Suggested repo structure

Mirror the pattern that already worked in `../test fyp/` — it kept data, scripts, model
artifacts, and outputs cleanly separated and made the eventual write-up straightforward:

```
code/
  data/
    primary/            # Malayalam (+ Tamil if in scope) political memes
    secondary/          # Tamil+Malayalam misogyny memes
    reference/           # Hateful Memes subsample
    splits/              # frozen 5-fold stratified split, shared by all arms
  arm1_classical/         # Swin-Tiny + BERT baseline
  arm2_vlm_benchmark/     # zero/few-shot/CoT across 4 VLMs
  arm3_peft_finetune/     # LoRA/QLoRA + ablation
  explainability/         # shared Grad-CAM + rationale + faithfulness module
  cross_lingual/          # checkpoint-transfer + prompt-transfer sub-flows
  evaluation/             # aggregation, comparison tables, ablation/faithfulness reports
  demo/                   # Gradio/Streamlit app
  outputs/                # predictions, logs, results — per arm subfolders
  instructions.md          # this file — keep updated as the plan evolves
```

---

## 8. Known risks (from the design review — don't rediscover these the hard way)

1. **Triplicated explainability logic** — fix: one shared module (Phase 5), not one per arm.
2. **LoRA/QLoRA applies asymmetrically** — Gemini can't be fine-tuned; give it a documented
   parallel prompt-optimization track instead of silently excluding it.
3. **Imbalance handling could silently diverge between arms** — fix: decide once at the
   Data layer (Phase 1), log which technique each arm used.
4. **Cross-lingual transfer is actually two flows, not one** — checkpoint transfer
   (Arms 1/3) vs. prompt-language transfer (Arm 2) — label them separately.
5. **"Best-performing model" for the demo is ambiguous** — fix: visible dropdown, no
   automatic selection.

---

## 9. When starting a fresh session from this file

- Read this file in full first.
- Read `../Team14_System_Design_Review.md` for the architectural reasoning behind the
  choices above.
- Read `../test fyp/README.md` and `../test fyp/Notes0.html` before touching Arm 3 or the
  data layer — both contain hard-won lessons (label normalization, image resize gotchas,
  Codabench's mislabeled download) that will otherwise get rediscovered the hard way on the
  secondary/reference datasets.
- Check off items in Section 6 as they're completed, and update Section 5's timeline
  assessment if the Mid Review date or team capacity changes.
