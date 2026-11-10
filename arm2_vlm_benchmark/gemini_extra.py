"""Worker A: extra Gemini 3.5 Flash-Lite runs (zero-shot-600, CoT, few-shot, Tamil, rationale).

Copied/adapted from gemini_zeroshot.py (same MODEL, same base INSTRUCTION_PROMPT, same
label parsing via structured JSON, same UNKNOWN handling = all retries failed).
Resumable: per-image results appended to outputs/gemini_extra/<task>.jsonl; reruns skip done ids.
API key read from .env.gemini (never printed).

Usage: python -m arm2_vlm_benchmark.gemini_extra <task> [--limit N]
tasks: zs600 | cot | fewshot | tamil | rationale
"""
import argparse, csv, json, os, random, sys, time
from google import genai
from google.genai import types
from PIL import Image

DATA_ROOT = "data/primary/malayalam"
OUT = "outputs/gemini_extra"
MODEL = "gemini-3.5-flash-lite"
TAMIL_CSV = "/Users/adithya/Final Year Project/test fyp/data/tamil/test.csv"

BASE = (
    "Analyze the political intent and target of this Malayalam political meme.\n\n"
    "INTENT: does the meme SUPPORT (praise) or CRITICISE (troll/oppose) its subject?\n"
    "TARGET: is the subject a specific PERSON, a PARTY/organization, or BOTH "
    "(the person and their party jointly)?\n\n"
    "Respond with your best judgment even if the meme's text is not in English."
)
# Tamil: same wording, language swapped. These are POLITICAL memes, not misogyny.
TAMIL = BASE.replace("Malayalam", "Tamil")
COT = BASE + ("\n\nThink step by step: first read the text in the image, say who or what is depicted, "
              "then reason about the tone and subject, and only then give the final intent and target.")
RAT = BASE + ("\n\nAlso give a one-sentence English rationale and the salient words/phrases (copied from "
              "the meme text, any script) or visual cues that drove your decision.")

S = types.Schema; T = types.Type
INTENT = S(type=T.STRING, enum=["SUPPORT", "CRITICISE"])
TARGET = S(type=T.STRING, enum=["PERSON", "PARTY", "BOTH"])
SCHEMA = S(type=T.OBJECT, properties={"intent": INTENT, "target": TARGET}, required=["intent", "target"])
SCHEMA_COT = S(type=T.OBJECT, properties={"reasoning": S(type=T.STRING), "intent": INTENT, "target": TARGET},
               required=["reasoning", "intent", "target"])
SCHEMA_RAT = S(type=T.OBJECT, properties={"rationale": S(type=T.STRING),
               "salient_words": S(type=T.ARRAY, items=S(type=T.STRING)), "intent": INTENT, "target": TARGET},
               required=["rationale", "salient_words", "intent", "target"])
I2L1 = {"SUPPORT": "support", "CRITICISE": "troll"}
T2L2 = {"PERSON": "person", "PARTY": "party", "BOTH": "intersection"}
L1_TO_INTENT = {v: k for k, v in I2L1.items()}
L2_TO_TARGET = {v: k for k, v in T2L2.items()}


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def malayalam(split):
    return [dict(image=r["image"], path=os.path.join(DATA_ROOT, r["image"]), split=split,
                 l1=r["level1"], l2=r["level2"]) for r in read_csv(os.path.join(DATA_ROOT, f"{split}.csv"))]


def fewshot_exemplars(seed=42):
    """1 train image per (level1,level2) combination, fixed seed, train only."""
    train = malayalam("train")
    rng = random.Random(seed)
    combos = sorted({(r["l1"], r["l2"]) for r in train})
    ex = [rng.choice([r for r in train if (r["l1"], r["l2"]) == c]) for c in combos]
    rng.shuffle(ex)
    assert all(e["split"] == "train" and "/test/" not in e["image"] for e in ex)
    return ex


def call(client, contents, schema, max_retries=12):
    last = None
    for a in range(max_retries):
        try:
            r = client.models.generate_content(model=MODEL, contents=contents, config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=schema, temperature=0))
            d = json.loads(r.text)
            return d, I2L1[d["intent"]], T2L2[d["target"]], None
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            wait = min(90, 5 * 2 ** a)
            print(f"  retry {a+1}/{max_retries}: {last[:120]} (wait {wait}s)", flush=True)
            time.sleep(wait)
    return None, "UNKNOWN", "UNKNOWN", last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("--limit", type=int); ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    key = None
    for line in open(".env.gemini"):
        if line.startswith("GEMINI_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"\'')
    if not key:
        sys.exit("no key in .env.gemini")
    client = genai.Client(api_key=key)

    prefix = []
    meta = {}
    if a.task == "zs600":
        rows = malayalam("train") + malayalam("test"); prompt, schema = BASE, SCHEMA
    elif a.task == "cot":
        rows = malayalam("test"); prompt, schema = COT, SCHEMA_COT
    elif a.task == "rationale":
        rows = malayalam("test"); prompt, schema = RAT, SCHEMA_RAT
    elif a.task == "fewshot":
        rows = malayalam("test"); schema = SCHEMA
        ex = fewshot_exemplars()
        prompt = BASE + "\n\nHere are labelled examples first, then the meme to classify."
        for i, e in enumerate(ex):
            prefix += [f"Example {i+1}:", Image.open(e["path"]).convert("RGB"),
                       json.dumps({"intent": L1_TO_INTENT[e["l1"]], "target": L2_TO_TARGET[e["l2"]]})]
        prefix.append("Now classify this meme:")
        ids = [e["image"] for e in ex]
        assert not set(ids) & {r["image"] for r in rows}
        meta = {"exemplars": [{"image": e["image"], "level1": e["l1"], "level2": e["l2"]} for e in ex],
                "test_ids_in_prompt": 0}
        json.dump(meta, open(f"{OUT}/fewshot_exemplars.json", "w"), indent=2)
    elif a.task == "tamil":
        rows = [dict(image=r["image"], path=r["image"], split="tamil", l1=r["level1"], l2=r["level2"])
                for r in read_csv(TAMIL_CSV)]; prompt, schema = TAMIL, SCHEMA
    else:
        sys.exit("unknown task")
    if a.limit:
        rows = rows[:a.limit]
    name = a.task + a.suffix
    jl = f"{OUT}/{name}.jsonl"
    done = set()
    if os.path.exists(jl):
        done = {json.loads(l)["image"] for l in open(jl) if l.strip()}
    with open(jl, "a") as f:
        for i, r in enumerate(rows):
            if r["image"] in done:
                continue
            img = Image.open(r["path"]).convert("RGB")
            d, p1, p2, err = call(client, prefix + [img, prompt], schema)
            if err:  # API failure after all retries: NOT a model UNKNOWN; do not persist, rerun later
                open(f"{OUT}/api_errors.log", "a").write(f"{name}\t{r['image']}\t{err[:80]}\n")
                print(f"[{i+1}/{len(rows)}] API_ERROR (not recorded, rerun to fill)", flush=True)
                continue
            rec = dict(image=r["image"], split=r["split"], true_level1=r["l1"], pred_level1=p1,
                       true_level2=r["l2"], pred_level2=p2, raw=d, error=err)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{i+1}/{len(rows)}] {r['image'][-14:]} true=({r['l1']},{r['l2']}) pred=({p1},{p2})", flush=True)
            time.sleep(a.sleep)
    # CSV in gemini_zeroshot schema
    recs = [json.loads(l) for l in open(jl) if l.strip()]
    with open(f"{OUT}/{name}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "split", "true_level1", "pred_level1", "true_level2", "pred_level2"])
        for x in recs:
            w.writerow([x["image"], x["split"], x["true_level1"], x["pred_level1"], x["true_level2"], x["pred_level2"]])
    print("done", name, len(recs), "UNKNOWN:", sum("UNKNOWN" in (x["pred_level1"], x["pred_level2"]) for x in recs))


if __name__ == "__main__":
    main()
