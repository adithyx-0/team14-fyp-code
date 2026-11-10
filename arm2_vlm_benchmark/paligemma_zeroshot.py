"""Arm 2 (local) — PaliGemma2-3B-mix-448 (8-bit MLX), zero-shot, Malayalam 100-image test.
Same prompt intent, parsing, UNKNOWN convention, image dir (512px copies) and output
JSON schema as local_vlm.py (imported for parse_output / label maps; local_vlm.py unchanged).
Resumable: per-image results cached in <out>.partial.json.

    "../test fyp/.venv/bin/python" -m arm2_vlm_benchmark.paligemma_zeroshot [--limit N] [--prompt v1]
"""
import argparse, csv, json, time
from pathlib import Path
from mlx_vlm import apply_chat_template, generate, load
from arm2_vlm_benchmark.local_vlm import (IMAGE_DIR, TEST_CSV, OUT_DIR, TASK, FORMAT, TO_LEVEL1, TO_LEVEL2, parse_output)

MODEL = "mlx-community/paligemma2-3b-mix-448-8bit"
PROMPTS = {"v1": TASK + FORMAT}  # identical wording to Qwen zero-shot, written once
# v2: PaliGemma-mix refuses long instructions ("not trained to answer"), so the same two
# questions are asked as two short single-turn prompts (one call each). Written once; the
# smoke test only checked parseability, not accuracy.
# v3: as v2 but INTENT asked in plain words (v2 gave "unanswerable" on 3/3 smoke images);
# answers mapped praise->SUPPORT, mock->CRITICISE before the shared parser.
Q3 = {"INTENT": "Does this meme praise or mock its subject? Answer with one word: praise or mock.", "TARGET": None}
INTENT_MAP = {"praise": "SUPPORT", "mock": "CRITICISE"}
Q2 = {"INTENT": "Does this political meme SUPPORT or CRITICISE its subject? Answer with one word: SUPPORT or CRITICISE.",
      "TARGET": "Is the subject of this political meme a PERSON, a PARTY, or BOTH? Answer with one word: PERSON, PARTY or BOTH."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--prompt", default="v1"); ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(TEST_CSV)))
    if a.limit: rows = rows[: a.limit]
    out = a.out or Path("outputs/paligemma") / f"paligemma2_3b_zeroshot_{a.prompt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.json")
    done = {r["image"]: r for r in json.load(open(part))} if part.exists() else {}
    model, processor = load(MODEL)
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS.get(a.prompt, "")}]}]
    try:
        prompt = apply_chat_template(processor, model.config, conv, num_images=1)
    except Exception as e:
        print("chat template failed, using raw prompt:", repr(e)[:100]); prompt = PROMPTS.get(a.prompt, "")
    t0 = time.time()
    for i, row in enumerate(rows):
        ip = IMAGE_DIR / Path(row["image"]).name
        if str(ip) in done: continue
        if a.prompt in ("v2", "v3"):
            outs = []
            qs = dict(Q2)
            if a.prompt == "v3": qs["INTENT"] = Q3["INTENT"]
            for k, q in qs.items():
                pr = apply_chat_template(processor, model.config, [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], num_images=1)
                rr = generate(model, processor, pr, image=str(ip), temperature=0.0, max_tokens=12, verbose=False)
                ans = (rr.text if hasattr(rr, "text") else str(rr)).strip()
                if a.prompt == "v3" and k == "INTENT":
                    ans = INTENT_MAP.get(ans.lower().strip(". "), ans)
                outs.append(f"{k}: " + ans)
            raw = "\n".join(outs)
        else:
            res = generate(model, processor, prompt, image=str(ip), temperature=0.0, max_tokens=40, verbose=False)
            raw = res.text if hasattr(res, "text") else str(res)
        p = parse_output(raw)
        done[str(ip)] = {"image": str(ip), "raw_output": raw, "gold_level1": row["level1"], "gold_level2": row["level2"],
                         "pred_level1": TO_LEVEL1.get(p["pred_level1"], "UNKNOWN"), "pred_level2": TO_LEVEL2.get(p["pred_level2"], "UNKNOWN")}
        json.dump(list(done.values()), open(part, "w"), ensure_ascii=False)
        r = done[str(ip)]; print(f"[{i+1}/{len(rows)}] {ip.name} raw={raw!r} pred=({r['pred_level1']},{r['pred_level2']})", flush=True)
    res = [done[str(IMAGE_DIR / Path(r["image"]).name)] for r in rows]
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    unk1 = sum(r["pred_level1"] == "UNKNOWN" for r in res); unk2 = sum(r["pred_level2"] == "UNKNOWN" for r in res)
    print(f"Saved {len(res)} to {out}; UNKNOWN L1 {unk1}, L2 {unk2}; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
