# System design — finalized for Mid Review

Per `HANDOFF.md` §4 (Adithya's task): pick one canonical diagram from the
three existing candidates, pair it with the LLD module table, and be
ready to defend Candidate B and the five risks in viva. This file is that
finalization — read alongside `../Team14_System_Design_Review.md` (the
full HLD/LLD reasoning) and `instructions.md` (the build plan).

---

## 1. Canonical diagram: `Overall_Architecture_Elaborated.png`

Three diagrams exist one level up from `code/`:

| Diagram | Verdict |
|---|---|
| `Overall_Architecture_Elaborated.png` | **Chosen — canonical.** |
| `System design v5.jpg` | Rejected — earlier/simpler draft, use only as a 10-second orientation slide if needed. |
| `refined_system_architecture_hld.svg` | Rejected — a clean minimal version of the same shape as the elaborated PNG, but strips the annotations that document the risk fixes. |

**Why the elaborated PNG wins:** it is the only one of the three that maps
box-for-box onto the LLD module table in `Team14_System_Design_Review.md`
*and* visually encodes all five risk fixes from that document's
self-critique, not just the happy-path architecture:

- A single **"Canonical data preparation"** box feeds all three arms —
  shows imbalance handling/splits are decided once upstream, not per arm
  (Risk 3).
- **Arm 3** carries the explicit sub-label *"Gemini: prompt-opt track, no
  fine-tune"* — the asymmetric-LoRA risk (Risk 2) is stated on the
  diagram itself, not left implicit.
- One shared **"Explainability and faithfulness layer"** box sits
  downstream of all three arms, with its own sub-boxes (Grad-CAM / NL
  rationale / faithfulness scorer) — the fix for Risk 1 (triplicated
  explainability logic) is drawn, not just described.
- **"Cross-lingual transfer"** is drawn as two explicitly labelled
  sub-boxes — *"Checkpoint transfer — Arms 1 and 3"* and
  *"Prompt-language transfer — Arm 2"* — directly fixing Risk 4 (it is
  not one uniform arrow).
- The **demo app** box has an explicit **"Model selector"** sub-box next
  to the Grad-CAM overlay and salient-token-highlight boxes — the fix for
  Risk 5 (no automatic "best model" selection) is visible on the diagram.

`System design v5.jpg` is a generic left-to-right pipeline swimlane — it
doesn't distinguish the shared explainability layer from the arms, has no
dataset provenance, and has no demo dropdown or split cross-lingual flow.
It reads as Candidate A with parallel arms squeezed into one stage, not
as the refined Candidate B. `refined_system_architecture_hld.svg` has the
right *shape* (same 8 boxes, same fan-out/fan-in) but drops the model
names and the risk-fix annotations — cleaner as a print handout, weaker
as a standalone defense artifact since a panel member reading it cold
won't see *why* it looks the way it does.

**Use in the deck:** one slide, `Overall_Architecture_Elaborated.png` in
full, immediately followed by the LLD table below — the two are
deliberately redundant with each other (the diagram is the table drawn
out), which is what makes them a strong paired pair rather than two
half-explanations.

---

## 2. Paired LLD module table

Reproduced from `Team14_System_Design_Review.md` §"Final refined design"
— present this table right after the diagram slide, not a different one:

| Module | Responsibility | Inputs | Outputs |
|---|---|---|---|
| Data & augmentation layer | Canonical 5-fold stratified split; three augmentation conditions (none / manual / LLM-synthetic) frozen once | Primary, secondary, reference datasets | Versioned dataset splits, per-condition training sets |
| Arm 1 — classical baseline | Frozen Swin-Tiny + BERT, residual fusion, 5-fold CV | Primary dataset splits | Anchor macro-F1 (both label levels), predictions, vision embeddings |
| Arm 2 — VLM benchmarking | Zero-shot / few-shot / CoT prompting across 4 VLMs; zero-shot pass on Hateful Memes for reference | Primary dataset, prompt templates, reference dataset | Predictions per model/strategy/label level, attention traces |
| Arm 3 — PEFT fine-tuning | LoRA/QLoRA (4-bit) on the 3 open-weight VLMs; rank/layer/seed ablation; Gemini's parallel track is prompt optimisation, not fine-tuning | Primary dataset, base VLM weights | Fine-tuned adapters/checkpoints, ablation results |
| Explainability & faithfulness layer | Grad-CAM extraction (Arm 1, and VLM vision towers where gradients are exposed), NL rationale generation (Arm 2/3), faithfulness scoring between the two | Predictions + embeddings/attention from all arms | Unified, type-aware explanation objects; faithfulness scores |
| Cross-lingual transfer | Checkpoint transfer sub-flow (Arms 1/3) and prompt-language transfer sub-flow (Arm 2), both on the secondary dataset only | Secondary dataset, checkpoints (1/3), prompts (2) | Stance/harm accuracy deltas, rationale-quality deltas across languages |
| Evaluation & comparison | Aggregates macro-F1 (both levels) across all arms, vs SYNAPSE (0.4256) and vs Hateful Memes reference, plus ablation and faithfulness results | Outputs of all arms + explainability + cross-lingual modules | Comparison tables and figures for the paper/report |
| Demo app | Accepts Malayalam/Tamil meme + optional caption; user selects which model to query | User input, any arm's model via a common inference call, explainability layer | Predicted label, confidence, Grad-CAM overlay, salient token highlights |

**Current build status against this table**, for viva honesty (don't
overclaim): Data layer and Arm 1 rows are fully built and reflect real
run output (`outputs/arm1_classical/`). Explainability row is
partially built — Grad-CAM half done for both label levels
(`outputs/explainability/`), NL-rationale half and faithfulness scorer
not started. Arm 2 row is scaffolded, not yet run (blocked on Adithyan's
API key). Arm 3, cross-lingual transfer, full evaluation aggregation, and
the demo app are explicitly deferred to post–Mid Review per the
scope-down decision in `mid_review_status.html` — say this proactively
rather than waiting to be asked.

---

## 3. Viva rehearsal — defending Candidate B

**Q: Why parallel arms instead of the sequential pipeline you originally
submitted a timeline for (Candidate A)?**
A: Candidate A is safe but architecturally inert — it's literally the
submitted timeline redrawn as boxes, and it blocks two of three teammates
at any given time since each stage gates the next. It also buries the
project's actual novelty claim (systematic multi-model comparison) as a
sub-step inside one stage instead of making it the structural centre of
the system. Candidate B fans out to three independent arms against one
frozen, versioned dataset — every teammate has an arm from day one, and
the comparison the paper is about *is* the architecture, not an
afterthought bolted onto it.

**Q: Candidate C (layered service) scores as well or better on almost
every row of your own decision matrix — extensibility, zero duplicated
explainability logic. Why not build that instead?**
A: Because those strengths cost real engineering time we don't have.
Forcing Gemini's API, three different local VLM inference stacks, and a
frozen CNN+BERT classifier behind one common `predict()`/`explain()`
interface is a real design problem on its own — an API model can't expose
gradients for Grad-CAM the way a local model can, so that contract has to
be honest about the differences before a single experiment can run. With
Mid Review inside a ~3-day runway at the time of this decision, that
upfront abstraction cost is exactly the "enterprise complexity" the
brief warns against introducing without need.

**Q: So you just picked B and ignored C's advantages?**
A: No — the self-critique in the design review found Candidate B's own
weak point is real: each arm implementing Grad-CAM independently risks
three incompatible output formats, which would break the one faithfulness
check the project has promised. The fix pulled *only* Candidate C's
narrowest useful idea — one shared, cross-cutting explainability &
faithfulness layer — into B, without adopting C's full plugin-registry
interface across all four heterogeneous model types. That hybrid is what
`Overall_Architecture_Elaborated.png` actually draws.

---

## 4. Viva rehearsal — the five risks and fixes

State these unprompted if the panel asks "what could go wrong with this
design" — it's stronger to lead with a self-critique than to wait for it:

1. **Triplicated explainability logic.** Each arm implementing its own
   Grad-CAM/rationale extraction risks three incompatible formats,
   undermining the promised faithfulness check. *Fix:* one shared module
   all arms call into, with a type-aware schema (`image_regions` /
   `salient_tokens`, both optional) feeding one faithfulness scorer.
2. **LoRA/QLoRA applies asymmetrically.** Gemini is API-only and can't be
   fine-tuned, so "Arm 3 = fine-tune the VLMs" silently drops one of four
   models from a track a panel will expect to see treated consistently.
   *Fix:* document it as a designed constraint and give Gemini its own
   parallel adaptation track — heavier few-shot/CoT prompt optimisation —
   so all four models still have an adapted variant, just via different
   mechanisms.
3. **Imbalance handling could silently diverge between arms.** With 23
   minority-class examples train-wide, each arm could independently pick
   an incompatible strategy, breaking apples-to-apples comparison. *Fix:*
   decide once at the Data layer, log which technique each arm actually
   used. (Concretely for Mid Review: augmentation itself deferred to
   Phase II per `data/DECISIONS.md`; Arm 1 uses class-weighted loss,
   logged there.)
4. **Cross-lingual transfer is described as one flow but is actually
   two.** "Transfer a checkpoint" only makes sense for Arms 1/3; Arm 2's
   zero/few-shot VLMs have no checkpoint to transfer. *Fix:* label two
   explicit sub-flows — checkpoint transfer (1/3) vs. prompt-language
   transfer (2) — so the HLD doesn't imply a uniform mechanism that
   doesn't exist.
5. **"Best-performing model" for the demo is ambiguous.** Three arms ×
   two label levels × two languages means "best" depends on which metric
   and class you weight — silently auto-picking one pre-empts the
   comparison the paper is supposed to make. *Fix:* visible model-choice
   dropdown in the demo, no automatic selection logic at all.

---

## 5. Viva rehearsal — tech-stack choices

**Why Swin-Tiny and not a larger CNN (ResNet-50, ConvNeXt) for Arm 1?**
Both backbones in Arm 1 are frozen (`arm1_classical/model.py` —
`requires_grad=False` on all vision/text params; only the fusion head and
two classification heads train). With 500 primary-dataset train examples,
that's the deliberate move: a bigger backbone doesn't help when it isn't
being fine-tuned, and a small trainable head on top of strong pretrained
features is far less prone to overfitting on 500 rows than training (or
even lightly fine-tuning) a larger vision model would be. Swin-Tiny also
gives a clean Grad-CAM target — `forward_features()` exposes a 7x7 spatial
grid analogous to a CNN's last conv layer (see
`explainability/gradcam_arm1.py` docstring) — without the extra
engineering a hierarchical-but-larger Swin variant would add for the same
frozen-feature use case.

**Why MuRIL and not mBERT for the text branch?**
MuRIL (`google/muril-base-cased`) is pretrained specifically on 17 Indian
languages including Malayalam, with explicit transliteration awareness —
a closer match to code-mixed, OCR-extracted Malayalam meme text than
generic mBERT's broad 100+-language coverage, where Indian-language depth
is diluted. Same frozen-backbone logic as the vision side: 500 training
rows isn't enough to profitably fine-tune a full multilingual BERT
regardless of which one is picked, so the choice is about which frozen
pretrained representation already fits this text distribution best.

**Why LoRA/QLoRA for Arm 3, not full fine-tuning?**
Three open-weight VLMs at 7B-class scale can't be fully fine-tuned on
realistically available hardware (a shared/booked college A100, per
`instructions.md` §4's open question on 40GB vs 80GB access) — and
`arm3_peft_finetune/` deliberately mirrors the SYNAPSE paper's own
LoRA config (rank=4, alpha=8, dropout=0.15, q_proj/v_proj only) so Arm 3's
numbers stay directly comparable to both the paper's published result and
this team's own SYNAPSE replication, not just internally consistent.
QLoRA's 4-bit quantization is what makes the rank/layer/seed ablation
grid (~30–40 runs) fit inside available GPU time at all.

**Gradio vs. Streamlit for the demo (open — not yet decided in code):**
`instructions.md` lists both as options and no demo code exists yet
(Phase 8, explicitly deferred to post–Mid Review). If asked in viva,
frame it as an open, low-stakes choice rather than a completed decision:
Gradio has slightly more purpose-built components for this exact demo
shape (image input, label + confidence output, image-overlay output) with
less custom layout code, which is why it's named first in the plan — but
nothing is blocked on deciding this before Mid Review.

---

## 6. What changed from the original design (say this if asked "did the
design evolve")

Per `Team14_System_Design_Review.md`'s own "what changed after the
critique" section — useful as evidence the design was actually
critiqued, not just accepted on first draft:
- Explainability: one shared module, not three.
- Gemini: explicit prompt-optimisation track instead of silent exclusion
  from "fine-tuning."
- Imbalance handling: fixed once at the Data layer, logged per arm.
- Cross-lingual transfer: two labelled sub-flows, not one.
- Demo: exposes model choice instead of implying a single "best" model.
