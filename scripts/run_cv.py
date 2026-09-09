#!/usr/bin/env python3
"""Run deterministic TRACE/PASE ablations for one or more CV folds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "classification_code" / "train_patchchestct_official_patch_fold0.py"
EVALUATOR = ROOT / "classification_code" / "evaluate_patchchestct_csea_raw_logits.py"

VARIANTS = {
    "baseline": {"pooling": "mean", "fine": 0.0, "gac": 0.0, "ramp": 0},
    "pase": {"pooling": "smooth-or", "fine": 0.0, "gac": 0.0, "ramp": 0},
    "pase-fine": {"pooling": "smooth-or", "fine": 0.25, "gac": 0.0, "ramp": 5},
    "trace": {"pooling": "smooth-or", "fine": 0.25, "gac": 0.05, "ramp": 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="trace")
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--gpu", default="0", help="Physical CUDA index exposed to each job")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", help="Default: <variant>_seed<seed>_e<epochs>")
    parser.add_argument(
        "--csea",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run fixed raw-logit CSEA (default: enabled for TRACE only)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checked_run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    args = parse_args()
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("--folds must contain integers from 0 to 4")
    spec = VARIANTS[args.variant]
    run_name = args.run_name or f"{args.variant}_seed{args.seed}_e{args.epochs}"
    use_csea = args.variant == "trace" if args.csea is None else args.csea

    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONHASHSEED": str(args.seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )

    for fold in args.folds:
        fold_dir = args.output_root.resolve() / run_name / f"fold_{fold}"
        if fold_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {fold_dir}")
        command = [
            sys.executable,
            "-u",
            str(TRAINER),
            "--splits-dir",
            str(args.splits_dir.resolve()),
            "--fold",
            str(fold),
            "--output-root",
            str(args.output_root.resolve()),
            "--run-name",
            run_name,
            "--backbone",
            "vjepa2_1_b",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "4",
            "--num-workers",
            str(args.num_workers),
            "--lr",
            "1e-5",
            "--head-lr-multiplier",
            "1",
            "--weight-decay",
            "1e-4",
            "--optimizer",
            "adamw",
            "--dice-weight",
            "1",
            "--patch-loss",
            "bce",
            "--mil-head",
            "basic",
            "--num-output-classes",
            "18",
            "--patch-grid-protocol",
            "anatomical_grid_v2_6x12x12",
            "--patch-token-pooling",
            str(spec["pooling"]),
            "--smooth-or-temperature",
            "1",
            "--fine-annotation-supervision-weight",
            str(spec["fine"]),
            "--fine-coarse-consistency-weight",
            str(spec["gac"]),
            "--fine-supervision-ramp-epochs",
            str(spec["ramp"]),
            "--checkpoint-metric",
            "val_loss",
            "--threshold-objective",
            "f1",
            "--patch-localization",
            "linear",
            "--seed",
            str(args.seed),
            "--gpu",
            str(args.gpu),
            "--device",
            "cuda",
            "--amp",
            "--deterministic",
        ]
        checked_run(command, env, args.dry_run)

        if use_csea:
            csea_dir = args.output_root.resolve() / f"{run_name}_csea_tau0p5" / f"fold_{fold}"
            if csea_dir.exists():
                raise FileExistsError(f"Refusing to overwrite CSEA output: {csea_dir}")
            csea_command = [
                sys.executable,
                "-u",
                str(EVALUATOR),
                "--source-run-dir",
                str(fold_dir),
                "--output-dir",
                str(csea_dir),
                "--device",
                "cuda",
                "--num-workers",
                str(args.num_workers),
            ]
            checked_run(csea_command, env, args.dry_run)


if __name__ == "__main__":
    main()
