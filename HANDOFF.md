# Handoff — Team 14, Mid Review prep (guide check-in Wed 19 Aug, Mid Review Fri 21 Aug)

Read `instructions.md` first for full project context. This file is just:
clone → set up → run your piece.

## 1. Get the code

```
git clone <REPO_URL>
cd code
```

## 2. Set up your environment

Requires Python 3.12 and `uv` (or plain `pip`, either works).

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate      # or: .venv/bin/python <script> without activating
```

This installs everything except the Arm 1 checkpoint (see below — too big for git).

## 3. If your task needs the Arm 1 checkpoint

`outputs/arm1_classical/final_full_train.pt` (~1GB) is **not in git** —
GitHub rejects files that size. If your task is the explainability
extension, you need this file specifically: ask Adithya to send it via
AirDrop/Drive (one-time, ~1GB). Put it at that exact path once you have it.

## 4. Your task

### If you're on Arm 2 (Gemini zero-shot)
1. Get a free API key: https://aistudio.google.com/apikey
2. `export GEMINI_API_KEY=...`
3. Smoke-test first (5 calls, not 100 — check output looks sane before spending quota):
   `python -m arm2_vlm_benchmark.gemini_zeroshot --limit 5`
4. If that looks right, run the full 100-image set:
   `python -m arm2_vlm_benchmark.gemini_zeroshot`
5. Results land in `outputs/arm2_vlm_benchmark/` — a predictions CSV and a
   results JSON with macro-F1 for both label levels, already lined up
   against Arm 1 / SYNAPSE paper / our SYNAPSE replication numbers.
6. Read the script's docstring (`arm2_vlm_benchmark/gemini_zeroshot.py`) —
   it explains two choices you should be ready to defend in viva: why the
   prompt is image-only (no OCR text), and why it reuses the SYNAPSE
   replication's own INTENT/TARGET label wording.
7. Once you have numbers, drop them into one slide: Gemini zero-shot vs
   Arm 1 vs SYNAPSE paper vs our replication (4 bars, both label levels).

### If you're on explainability + system design
1. Get `final_full_train.pt` from Adithya (see step 3 above).
2. Run: `python -m explainability.gradcam_arm1 --n 16 --level level2`
3. Look through `outputs/explainability/gradcam/` (level1 heatmaps already
   exist from today, level2 ones you just generated) — pick 3-4 that make
   a clean visual point (a correct prediction where the heatmap lands on
   something sensible, plus one wrong prediction, is a stronger slide than
   4 random ones).
4. Read `explainability/gradcam_arm1.py`'s docstring — explains why
   Grad-CAM works on a frozen Swin transformer the way it normally works
   on a CNN's last conv layer (spoiler: `forward_features()` gives a 7x7
   spatial grid to hook into instead).
5. For system design: open `Team14_System_Design_Review.md` (one level up
   from `code/`), pick one of the existing diagrams
   (`System design v5.jpg`, `Overall_Architecture_Elaborated.png`, or
   `refined_system_architecture_hld.svg`) as the canonical one, pair it
   with the LLD module table from that doc, and be ready to defend: why
   Candidate B (parallel arms) over the sequential/layered alternatives,
   and the 5 named risks-and-fixes.

### If you're on literature review
1. Open `GROUP 14 LITERATURE REVIEW FL.pdf` (one level up from `code/`).
2. Build a comparison table: rows = 4-5 most relevant papers (SYNAPSE plus
   whichever others are closest), columns = task / language / modality /
   explainability / fine-tuning / **this work**. This is reformatting
   material that already exists, not new research — should be a couple
   hours, not a literature search.

## 5. Before you push

- `git status` — make sure you're not accidentally adding `.venv/`,
  `__pycache__/`, or any `.pt` file (the `.gitignore` should already catch
  these, but double check).
- Commit + push to your own branch if you're unsure, or straight to `main`
  if it's a small, self-contained change (small team, short timeline —
  don't over-process this with PRs).
- `git pull` before you start each session so you're not working on a
  stale copy.

## 6. Tonight (all three)

Sync on the 5-minute walkthrough for tomorrow's guide check-in: Arm 1
result, Grad-CAM (already real), the scoped plan, and the two blockers to
explicitly flag for the guide — A100 access unconfirmed, and the
secondary/reference datasets were never requested (both need lead time,
not more hours).
