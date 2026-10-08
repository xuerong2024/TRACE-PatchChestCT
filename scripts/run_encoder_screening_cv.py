#!/usr/bin/env python3
"""Run the paper's five-fold case-level NoisyOR encoder screening."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper" / "encoder_screening.json"
TRAINER = ROOT / "classification_code" / "train_patchchestct_official_case_fold0.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--methods", nargs="+", help="Defaults to every method in --config")
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    shared = config["shared"]
    methods = config["methods"]
    if shared.get("encoder_training") != "end_to_end":
        raise ValueError("Paper encoder screening requires encoder_training='end_to_end'")
    if shared.get("case_label_source") != "manifest disease labels":
        raise ValueError("Paper encoder screening requires manifest disease labels")
    if shared.get("fold_standard_deviation") != "population":
        raise ValueError("Paper encoder screening requires population fold standard deviation")
    selected = args.methods or list(methods)
    unknown = sorted(set(selected) - set(methods))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; available methods: {sorted(methods)}")
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("--folds must contain integers from 0 to 4")

    seed = int(shared["seed"])
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )

    for method_name in selected:
        backbone, epochs, batch_size, accumulation, learning_rate = methods[method_name]
        run_name = f"{backbone}_case_official_seed{seed}_e{epochs}"
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
                str(backbone),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--gradient-accumulation-steps",
                str(accumulation),
                "--num-workers",
                str(args.num_workers or shared["num_workers"]),
                "--lr",
                str(learning_rate),
                "--weight-decay",
                str(shared["weight_decay"]),
                "--noisy-or-alpha",
                str(shared["noisy_or_alpha"]),
                "--checkpoint-metric",
                str(shared["checkpoint_metric"]),
                "--threshold-objective",
                str(shared["threshold_objective"]),
                "--seed",
                str(seed),
                "--gpu",
                str(args.gpu),
                "--device",
                "cuda",
                "--amp",
                "--deterministic",
            ]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
