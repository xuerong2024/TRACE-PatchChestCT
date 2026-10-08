#!/usr/bin/env python3
"""Run the paper's joint classification/localization main-table experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper" / "joint_cv.json"
TRAINER = ROOT / "classification_code" / "train_patchchestct_official_patch_fold0.py"
EVALUATOR = ROOT / "classification_code" / "evaluate_patchchestct_csea_raw_logits.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--methods", nargs="+", help="Defaults to every method in --config")
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--voco-pretrained-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checked_run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def option(command: list[str], name: str, value: Any) -> None:
    command.extend((name, str(value)))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    shared = config["shared"]
    methods = config["methods"]
    if shared.get("encoder_training") != "end_to_end":
        raise ValueError("Paper joint experiments require encoder_training='end_to_end'")
    if shared.get("fold_standard_deviation") != "population":
        raise ValueError("Paper joint experiments require population fold standard deviation")
    selected_methods = args.methods or list(methods)
    unknown_methods = sorted(set(selected_methods) - set(methods))
    if unknown_methods:
        raise ValueError(
            f"Unknown methods {unknown_methods}; available methods: {sorted(methods)}"
        )
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("--folds must contain integers from 0 to 4")
    if "voco10k" in selected_methods and args.voco_pretrained_checkpoint is None:
        raise ValueError("voco10k requires --voco-pretrained-checkpoint /path/to/VoCo_10k.pt")

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

    for method_name in selected_methods:
        spec = methods[method_name]
        for fold in args.folds:
            fold_dir = args.output_root.resolve() / spec["run_name"] / f"fold_{fold}"
            if fold_dir.exists():
                raise FileExistsError(f"Refusing to overwrite existing run: {fold_dir}")
            command = [sys.executable, "-u", str(TRAINER)]
            for flag, value in (
                ("--splits-dir", args.splits_dir.resolve()),
                ("--fold", fold),
                ("--output-root", args.output_root.resolve()),
                ("--run-name", spec["run_name"]),
                ("--backbone", spec["backbone"]),
                ("--epochs", spec["epochs"]),
                ("--batch-size", spec["batch_size"]),
                ("--gradient-accumulation-steps", spec["gradient_accumulation_steps"]),
                ("--num-workers", args.num_workers or shared["num_workers"]),
                ("--lr", spec["learning_rate"]),
                ("--head-lr-multiplier", 1),
                ("--weight-decay", shared["weight_decay"]),
                ("--optimizer", spec["optimizer"]),
                ("--dice-weight", shared["dice_weight"]),
                ("--patch-loss", shared["patch_loss"]),
                ("--mil-head", "basic"),
                ("--num-output-classes", 18),
                ("--patch-grid-protocol", shared["patch_grid_protocol"]),
                ("--patch-token-pooling", spec["patch_token_pooling"]),
                ("--smooth-or-temperature", spec.get("smooth_or_temperature", 1.0)),
                ("--fine-annotation-supervision-weight", spec.get("fine_annotation_supervision_weight", 0.0)),
                ("--fine-coarse-consistency-weight", spec.get("fine_coarse_consistency_weight", 0.0)),
                ("--fine-supervision-ramp-epochs", spec.get("fine_supervision_ramp_epochs", 0)),
                ("--checkpoint-metric", shared["checkpoint_metric"]),
                ("--threshold-objective", shared["threshold_objective"]),
                ("--patch-localization", spec.get("patch_localization", "linear")),
                ("--seed", seed),
                ("--gpu", args.gpu),
                ("--device", "cuda"),
            ):
                option(command, flag, value)
            if spec.get("requires_voco_checkpoint"):
                option(command, "--voco-pretrained-checkpoint", args.voco_pretrained_checkpoint.resolve())
            command.extend(("--amp", "--deterministic"))
            checked_run(command, env, args.dry_run)

            if spec.get("csea"):
                csea_dir = args.output_root.resolve() / f"{spec['run_name']}_csea_tau0p5" / f"fold_{fold}"
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
                    str(args.num_workers or shared["num_workers"]),
                ]
                checked_run(csea_command, env, args.dry_run)


if __name__ == "__main__":
    main()
