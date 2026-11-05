"""Arm 2 (local) — open-weight VLM, prompt-only (no weight updates), on the
Malayalam 100-image test set. Hardware-constrained stand-in for the planned
7B VLMs: Qwen3-VL-2B-Instruct (8-bit MLX) runs on the 16GB MacBook; the 7B
models need an A100 that is not available (see data/DECISIONS.md).

Same task wording and label vocabulary as `gemini_zeroshot.py` (INTENT:
SUPPORT/CRITICISE x TARGET: PERSON/PARTY/BOTH), and the same 100 test
images as Arm 1 / the SYNAPSE replication, so all arms are comparable.
Image-only prompting (no OCR text) for the same reason as the Gemini arm.

Two prompting strategies (greedy decoding, temperature 0):
  zeroshot  - task description + required answer format.
  cot       - Wei et al. 2022 style: brief step-by-step reasoning (who is
              shown, tone), capped at 120 tokens; then, as in Kojima et al.
              2022's two-stage answer extraction, a second turn asks only
              for the two answer lines. Needed because the 2B model's free
              reasoning often degenerates into repetition and exhausts the
              token budget before ever answering (observed in smoke test).
              Parsing reads the LAST INTENT:/TARGET: lines.

Runs under the replication's interpreter, which has mlx-vlm installed
(code/.venv does not, and mixing them risks transformers version clashes):

    "../test fyp/.venv/bin/python" -m arm2_vlm_benchmark.local_vlm \
        --strategy zeroshot [--limit 5]
"""
import argparse
import json
import re
import time
from pathlib import Path

from mlx_vlm import apply_chat_template, generate, load

MODEL_DIR = Path("../test fyp/models/Qwen3-VL-2B-Instruct-MLX-8bit")
# Resized (512px) copies used by the replication / LoRA training: the model's
# processor upsamples small images, so this keeps zero-shot and fine-tuned
# inputs identical. Test ids and labels were verified identical to
# data/primary/malayalam/test.csv (100/100).
IMAGE_DIR = Path("../test fyp/data/resized_512/malayalam_test_dataset")
TEST_CSV = Path("data/primary/malayalam/test.csv")
OUT_DIR = Path("outputs/arm2_vlm_benchmark")

VALID_L1 = {"SUPPORT", "CRITICISE"}
VALID_L2 = {"PERSON", "PARTY", "BOTH"}

TASK = (
    "Analyze the political intent and target of this Malayalam political meme.\n\n"
    "INTENT: does the meme SUPPORT (praise) or CRITICISE (troll/oppose) its subject?\n"
    "TARGET: is the subject a specific PERSON, a PARTY/organization, or BOTH "
    "(the person and their party jointly)?\n\n"
    "Respond with your best judgment even if the meme's text is not in English.\n\n"
)
FORMAT = "Answer in exactly this format:\nINTENT: <SUPPORT or CRITICISE>\nTARGET: <PERSON or PARTY or BOTH>"
PROMPTS = {
    "zeroshot": TASK + FORMAT,
    "cot": TASK
    + "Think step by step in at most 3 short sentences: who or what is shown and "
    "whether the tone praises or mocks them. Do not transcribe the text.",
}
EXTRACT = "Based on your reasoning, give your final answer.\n" + FORMAT
MAX_TOKENS = {"zeroshot": 40, "cot": 120}

TO_LEVEL1 = {"SUPPORT": "support", "CRITICISE": "troll"}
TO_LEVEL2 = {"PERSON": "person", "PARTY": "party", "BOTH": "intersection"}


def parse_output(text: str) -> dict:
    """Last valid INTENT:/TARGET: line wins; anything unparseable -> UNKNOWN."""
    intent, target = "UNKNOWN", "UNKNOWN"
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        m = re.match(r"INTENT:\s*\W*(\w+)", line, re.IGNORECASE)
        if m and m.group(1).upper() in VALID_L1:
            intent = m.group(1).upper()
        m = re.match(r"TARGET:\s*\W*(\w+)", line, re.IGNORECASE)
        if m and m.group(1).upper() in VALID_L2:
            target = m.group(1).upper()
    return {"pred_level1": intent, "pred_level2": target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=sorted(PROMPTS), default="zeroshot")
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import csv

    with open(TEST_CSV) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading {MODEL_DIR} ...")
    model, processor = load(str(MODEL_DIR), processor_config={"trust_remote_code": True})
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[args.strategy]}]}]
    prompt = apply_chat_template(processor, model.config, conversation, num_images=1)

    results, t0 = [], time.time()
    for i, row in enumerate(rows):
        image_path = IMAGE_DIR / Path(row["image"]).name
        if not image_path.exists():
            raise SystemExit(f"missing image: {image_path}")
        res = generate(model, processor, prompt, image=str(image_path),
                       temperature=0.0, max_tokens=MAX_TOKENS[args.strategy], verbose=False)
        raw = res.text if hasattr(res, "text") else str(res)
        if args.strategy == "cot":
            followup = conversation + [
                {"role": "assistant", "content": [{"type": "text", "text": raw}]},
                {"role": "user", "content": [{"type": "text", "text": EXTRACT}]},
            ]
            p2 = apply_chat_template(processor, model.config, followup, num_images=1)
            res2 = generate(model, processor, p2, image=str(image_path),
                            temperature=0.0, max_tokens=24, verbose=False)
            raw = raw + "\n---\n" + (res2.text if hasattr(res2, "text") else str(res2))
        parsed = parse_output(raw)
        results.append({
            "image": str(image_path),
            "raw_output": raw,
            "gold_level1": row["level1"],
            "gold_level2": row["level2"],
            "pred_level1": TO_LEVEL1.get(parsed["pred_level1"], "UNKNOWN"),
            "pred_level2": TO_LEVEL2.get(parsed["pred_level2"], "UNKNOWN"),
        })
        r = results[-1]
        print(f"[{i+1}/{len(rows)}] {image_path.name}: pred=({r['pred_level1']},{r['pred_level2']}) "
              f"gold=({r['gold_level1']},{r['gold_level2']})", flush=True)

    out = args.out or OUT_DIR / f"qwen3vl2b_{args.strategy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    unk = sum(r["pred_level1"] == "UNKNOWN" or r["pred_level2"] == "UNKNOWN" for r in results)
    print(f"Saved {len(results)} predictions to {out} ({unk} unparsed) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
