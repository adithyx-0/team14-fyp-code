"""Arm 2 — Gemini 2.5 Flash, zero-shot, on the primary Malayalam test set.

Scope (per instructions.md Phase 3 / mid_review_status.html "must-do #1"):
zero-shot only, no few-shot/CoT yet, on the same 100-image held-out test
set Arm 1 was scored on, so the two arms are directly comparable.

Image-only prompting (no OCR text passed in) — deliberate, not an
oversight: Arm 1's whole design forces OCR as a separate text-branch input
because its encoders can't read pixels; a VLM's actual selling point is
reading the meme's embedded text natively from the image. Feeding it OCR
text on top would blur what's being benchmarked. State this choice if
asked in viva.

Reuses the exact instruction framing and label vocabulary from the
SYNAPSE replication's own prompt (`../test fyp/scripts/prepare_dataset.py`,
`infer.py`) — same INTENT (SUPPORT/CRITICISE) x TARGET (PERSON/PARTY/BOTH)
scheme — so this arm's zero-shot number is conceptually comparable to the
paper's own labels, not a differently-worded task. Parsing uses Gemini's
structured JSON output mode instead of the replication's regex-on-free-text,
since that's simply more reliable when available, not a scope difference.

Setup:
    export GEMINI_API_KEY=...     # https://aistudio.google.com/apikey
    python -m arm2_vlm_benchmark.gemini_zeroshot --limit 5   # smoke test first
    python -m arm2_vlm_benchmark.gemini_zeroshot             # full 100-image run
"""
import argparse
import csv
import json
import os
import time

from google import genai
from google.genai import types
from PIL import Image
from sklearn.metrics import f1_score

DATA_ROOT = "data/primary/malayalam"
TEST_OCR_CSV = os.path.join(DATA_ROOT, "test_ocr.csv")  # only level1/level2/image cols used
OUT_DIR = "outputs/arm2_vlm_benchmark"
MODEL = "gemini-2.5-flash"

INSTRUCTION_PROMPT = (
    "Analyze the political intent and target of this Malayalam political meme.\n\n"
    "INTENT: does the meme SUPPORT (praise) or CRITICISE (troll/oppose) its subject?\n"
    "TARGET: is the subject a specific PERSON, a PARTY/organization, or BOTH "
    "(the person and their party jointly)?\n\n"
    "Respond with your best judgment even if the meme's text is not in English."
)

# JSON schema for structured output — mirrors the SYNAPSE replication's own
# INTENT/TARGET label vocabulary (see module docstring)
RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "intent": types.Schema(type=types.Type.STRING, enum=["SUPPORT", "CRITICISE"]),
        "target": types.Schema(type=types.Type.STRING, enum=["PERSON", "PARTY", "BOTH"]),
    },
    required=["intent", "target"],
)

INTENT_TO_LEVEL1 = {"SUPPORT": "support", "CRITICISE": "troll"}
TARGET_TO_LEVEL2 = {"PERSON": "person", "PARTY": "party", "BOTH": "intersection"}


def classify_one(client, image_path, max_retries=3):
    img = Image.open(image_path).convert("RGB")
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[img, INSTRUCTION_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0,  # greedy decoding, matches the replication's own inference setup
                ),
            )
            data = json.loads(resp.text)
            return INTENT_TO_LEVEL1[data["intent"]], TARGET_TO_LEVEL2[data["target"]]
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/{max_retries} after error: {e} (waiting {wait}s)")
            time.sleep(wait)
    return "UNKNOWN", "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run on only the first N rows (smoke test)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between API calls (free-tier rate limit)")
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey "
            "then: export GEMINI_API_KEY=..."
        )

    client = genai.Client()  # reads GEMINI_API_KEY from env automatically

    with open(TEST_OCR_CSV) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    pred_rows = []
    for i, row in enumerate(rows):
        image_path = os.path.join(DATA_ROOT, row["image"])
        pred_l1, pred_l2 = classify_one(client, image_path)
        pred_rows.append(
            {
                "image": row["image"],
                "true_level1": row["level1"],
                "pred_level1": pred_l1,
                "true_level2": row["level2"],
                "pred_level2": pred_l2,
            }
        )
        print(f"[{i+1}/{len(rows)}] {row['image']}: true=({row['level1']},{row['level2']}) pred=({pred_l1},{pred_l2})")
        time.sleep(args.sleep)

    pred_csv_path = os.path.join(OUT_DIR, "gemini_zeroshot_predictions.csv")
    with open(pred_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    # UNKNOWN predictions (all retries failed) count as wrong, not excluded —
    # an API failure is a real failure mode for this arm, not something to
    # quietly drop from the metric
    f1_l1 = f1_score([r["true_level1"] for r in pred_rows], [r["pred_level1"] for r in pred_rows], average="macro", zero_division=0)
    f1_l2 = f1_score([r["true_level2"] for r in pred_rows], [r["pred_level2"] for r in pred_rows], average="macro", zero_division=0)

    results = {
        "n_examples": len(pred_rows),
        "n_unknown": sum(1 for r in pred_rows if "UNKNOWN" in (r["pred_level1"], r["pred_level2"])),
        "gemini_zeroshot_f1_level1": f1_l1,
        "gemini_zeroshot_f1_level2": f1_l2,
        "arm1_test_f1_level1": 0.5076,
        "arm1_test_f1_level2": 0.3940,
        "synapse_paper_test_f1_level2": 0.4256,
        "synapse_replication_test_f1_level2": 0.3183,
    }
    with open(os.path.join(OUT_DIR, "gemini_zeroshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGemini 2.5 Flash zero-shot: level1 macro-F1={f1_l1:.4f} level2 macro-F1={f1_l2:.4f}")
    print(f"predictions -> {pred_csv_path}")
    print(f"results -> {os.path.join(OUT_DIR, 'gemini_zeroshot_results.json')}")


if __name__ == "__main__":
    main()
