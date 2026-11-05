"""
Arm 3 trainer: copy of ../test fyp/scripts/train_lora.py plus a --seed flag
(python/numpy/mlx RNGs) so the ablation can measure seed variance. Run from
`../test fyp` with that directory's interpreter (see run_ablation.py).

LoRA fine-tuning driver mirroring SYNAPSE (Suriya et al., DravidianLangTech
@ACL 2026, section 4.3): LoRA applied to q_proj/v_proj of the language-model
decoder only (vision tower + rest of the model frozen), rank=4, alpha=8,
dropout=0.15, batch size 1 with gradient accumulation 8 (effective batch 8),
lr=5e-5, weight decay 0.1, 500 steps.

We drive mlx-vlm's own SFT trainer internals directly (rather than the
`python -m mlx_vlm.lora` CLI) so we can load a locally-prepared dataset via
`datasets.load_from_disk` instead of the CLI's hub-only `load_dataset` call.

Usage:
    python scripts/train_lora.py \
        --model models/Qwen3-VL-2B-Instruct-MLX-8bit \
        --train-dataset data/train_dataset \
        --out adapters/synapse_lora
"""

import argparse
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx.optimizers as optim
from datasets import load_from_disk

from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer import sft_trainer
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.trainer.utils import (
    find_all_linear_names,
    get_peft_model,
    print_trainable_parameters,
)
from mlx_vlm.utils import load


def _safe_collate_grid_thw(values):
    """Replacement for mlx_vlm.trainer.sft_trainer._collate_grid_thw.

    The stock version does mx.array(value) on the processor's grid array and
    intermittently yields garbage (observed t=34359738368 = 2**35), which
    then crashes get_rope_index with `TypeError: arange(): incompatible
    function arguments`. Seen in the unmodified replication trainer too, so
    unrelated to seeding. Fix: force an explicit contiguous int32 copy via
    numpy, and fail fast on implausible values (the driver retries a failed
    run) instead of training on corrupted positions.
    """
    rows = []
    for value in values:
        arr = np.array(value).astype(np.int32).reshape(-1, 3)
        if arr.min() < 1 or arr.max() > 4096:
            raise ValueError(f"implausible image_grid_thw {arr.tolist()}")
        rows.append(arr)
    return mx.array(np.concatenate(rows, axis=0))


sft_trainer._collate_grid_thw = _safe_collate_grid_thw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--steps-per-report", type=int, default=5)
    parser.add_argument("--steps-per-save", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mem-limit-gb", type=float, default=8.0, help="cap MLX memory so the Mac stays usable")
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    args = parser.parse_args()
    mx.set_memory_limit(int(args.mem_limit_gb * 1024**3))
    mx.set_cache_limit(512 * 1024**2)  # freed buffers are not hoarded
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)  # LoRA init + dropout masks

    args.out.mkdir(parents=True, exist_ok=True)
    adapter_file = args.out / "adapters.safetensors"

    print(f"Loading base model from {args.model} ...")
    model, processor = load(str(args.model), processor_config={"trust_remote_code": True})
    config = model.config.__dict__

    print(f"Loading dataset from {args.train_dataset} ...")
    hf_dataset = load_from_disk(str(args.train_dataset))
    train_dataset = VisionDataset(hf_dataset, config, processor, train_on_completions=True)

    print(f"Applying LoRA to {args.target_modules} (rank={args.rank}, alpha={args.alpha}, dropout={args.dropout}) ...")
    modules = [m for m in args.target_modules if m in find_all_linear_names(model.language_model)]
    if not modules:
        raise SystemExit(
            f"None of {args.target_modules} found among linear layers: "
            f"{find_all_linear_names(model.language_model)}"
        )
    model = get_peft_model(model, modules, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    print_trainable_parameters(model.language_model)

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        steps_per_report=args.steps_per_report,
        steps_per_eval=max(args.iters, 10**9),  # no held-out val split for the smoke test
        steps_per_save=args.steps_per_save,
        adapter_file=str(adapter_file),
        grad_checkpoint=True,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
    )

    print(f"Training for {args.iters} steps (effective batch = {args.batch_size * args.grad_accum}) ...")
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
        train_on_completions=True,
        assistant_id=None,
    )
    print(f"Done. Adapter saved to {adapter_file}")


if __name__ == "__main__":
    main()
