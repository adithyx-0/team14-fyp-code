"""Arm 3 — LoRA fine-tuning ablation for Qwen3-VL-2B-Instruct (8-bit MLX) on
the Malayalam primary dataset, run on the MacBook.

Hardware-constrained deviation (logged in data/DECISIONS.md): the plan
targets 7B VLMs with QLoRA on an A100 and a 30-40 run grid. Neither is
available, so this fine-tunes the 2B model the SYNAPSE replication already
validated (same LoRA recipe as the paper, section 4.3: alpha = 2*rank,
dropout 0.15, q_proj/v_proj only, lr 5e-5, wd 0.1, effective batch 8, 500
steps) and runs a reduced, resumable grid that varies exactly the factors
the plan names, each with a reason:

  r4_s42 / r4_s43 / r4_s44   paper's rank over 3 seeds -> run-to-run noise,
                             needed to judge whether any other difference
                             is real (test set is only 100 examples).
  r8_s42 / r16_s42           rank ablation (plan: "rank" axis).
  r4_os_s42                  rank 4 + minority-class oversampling of the
                             training set (party x2, intersection x4): the
                             replication never predicts BOTH (0% recall),
                             the Risk-3 imbalance question. Oversampling
                             is a training-data change only; the test set
                             is untouched.

Every run is skipped if its predictions file already exists, so the script
can be interrupted and restarted. Execute under the replication's
interpreter (it has mlx-vlm; code/.venv does not):

    "../test fyp/.venv/bin/python" -m arm3_peft_finetune.run_ablation [--only NAME ...]
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
REPL = (CODE / ".." / "test fyp").resolve()
MODEL = REPL / "models" / "Qwen3-VL-2B-Instruct-MLX-8bit"
TRAIN_DS = "data/malayalam_train_dataset"          # relative to REPL (image paths inside are relative too)
TRAIN_OS_DS = "data/malayalam_train_oversampled_dataset"
TEST_DS = "data/malayalam_test_dataset"
OUT = CODE / "outputs" / "arm3_peft_finetune"

OVERSAMPLE = {"party": 2, "intersection": 4}  # repeat factors; person (majority) x1

GRID = [
    # name,      rank, seed, oversample
    ("r4_s42",   4,  42, False),
    ("r8_s42",   8,  42, False),
    ("r16_s42",  16, 42, False),
    ("r4_os_s42", 4, 42, True),
    ("r4_s43",   4,  43, False),
    ("r4_s44",   4,  44, False),
    # seed repeats of the oversampling arm: is the r4_os_s42 gain real or one lucky seed?
    ("r4_os_s43", 4, 43, True),
    ("r4_os_s44", 4, 44, True),
    # extra seeds so rank 8 / rank 16 can be judged by a seed mean, like rank 4
    ("r8_s43",   8,  43, False),
    ("r8_s44",   8,  44, False),
    ("r16_s43",  16, 43, False),
    ("r16_s44",  16, 44, False),
]


def _has_nan_loss(log):
    """True if any logged training loss is NaN (a corrupted run must be redone, never scored)."""
    import re
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(log).read_text(errors="ignore"))
    return re.search(r"Train loss\s+(nan|inf)", text, re.IGNORECASE) is not None


def run(cmd, log, attempts=4, reject_nan=False):
    """Run a subprocess, retrying from scratch on failure (the mlx-vlm
    Qwen3-VL trainer is intermittently flaky; see train_lora.py)."""
    for attempt in range(1, attempts + 1):
        with open(log, "w") as f:
            r = subprocess.run([str(c) for c in cmd], cwd=REPL, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            if not (reject_nan and _has_nan_loss(log)):
                return
            print(f"  attempt {attempt}/{attempts}: NaN training loss, discarding run", flush=True)
            continue
        print(f"  attempt {attempt}/{attempts} failed (see {log})", flush=True)
    raise SystemExit(f"command failed {attempts}x (see {log}): {' '.join(map(str, cmd))}")


def ensure_oversampled_dataset():
    if (REPL / TRAIN_OS_DS).exists():
        return
    with open(REPL / "data/malayalam/train.csv") as f:
        rows = list(csv.DictReader(f))
    out_rows = [r for r in rows for _ in range(OVERSAMPLE.get(r["level2"].strip().lower(), 1))]
    csv_path = OUT / "train_oversampled.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    print(f"oversampled train set: {len(rows)} -> {len(out_rows)} rows", flush=True)
    run([sys.executable, "scripts/prepare_dataset.py", "--csv", csv_path, "--out", TRAIN_OS_DS,
         "--language", "malayalam"], OUT / "logs" / "prepare_oversampled.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="run only these config names")
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()

    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for name, rank, seed, os_ in GRID:
        if args.only and name not in args.only:
            continue
        pred = OUT / f"{name}.json"
        if pred.exists():
            print(f"[skip] {name}: {pred.name} exists", flush=True)
            continue
        if os_:
            ensure_oversampled_dataset()
        adapter_dir = OUT / "adapters" / name
        t0 = time.time()
        print(f"[train] {name} rank={rank} seed={seed} oversample={os_}", flush=True)
        run([sys.executable, CODE / "arm3_peft_finetune" / "train_lora.py",
             "--model", MODEL, "--train-dataset", TRAIN_OS_DS if os_ else TRAIN_DS,
             "--out", adapter_dir, "--rank", rank, "--alpha", 2 * rank, "--seed", seed,
             "--iters", args.iters], OUT / "logs" / f"{name}_train.log", reject_nan=True)
        print(f"[infer] {name} (train took {time.time()-t0:.0f}s)", flush=True)
        run([sys.executable, "scripts/infer.py", "--model", MODEL, "--adapter", adapter_dir,
             "--test-dataset", TEST_DS, "--out", pred, "--language", "malayalam"],
            OUT / "logs" / f"{name}_infer.log")
        print(f"[done] {name} in {time.time()-t0:.0f}s -> {pred}", flush=True)


if __name__ == "__main__":
    main()
