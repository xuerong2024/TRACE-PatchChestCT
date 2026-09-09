#!/usr/bin/env python3
"""Formally evaluate pre-locked CSEA from raw V-JEPA patch logits.

This evaluator is deliberately independent of training.  It loads a completed
best checkpoint, reproduces the source max-pooling case predictions as an
invariance check, and then replaces only the case readout with

    sigmoid(LogMeanExp(coarse_patch_logits, tau=0.5)).

The CSEA temperature is a module constant rather than a command-line option so
the formal run cannot silently sweep it on validation or test data.  Case
thresholds are selected per class on validation by F1 and then frozen for the
test set.

Localization is intentionally unaffected by CSEA:

* checkpoints without a fine branch use sigmoid(direct coarse logits);
* Fine and Fine+GAC checkpoints use sigmoid(SmoothOR(fine logits -> 6x12x12))
  with the source training temperature.

Both the historical test-oracle Patch-DSC and a leakage-free
Patch-DSC@validation-selected-threshold are saved.  Existing run directories
are read only, and an existing output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classification_code import train_patchchestct_official_patch_fold0 as trainmod  # noqa: E402
from classification_code.patchchestct_grid import (  # noqa: E402
    ANATOMICAL_GRID_V2,
    reduce_patch_target_24_to_6,
)
from classification_code.patchchestct_pooling import smooth_logmeanexp_pool3d  # noqa: E402


CSEA_TEMPERATURE = 0.5
CSEA_OUTPUT_SHAPE = (1, 1, 1)
COARSE_GRID = (6, 12, 12)
FINE_GRID = (24, 12, 12)
CASE_THRESHOLD_OBJECTIVE = "f1"
LOCALIZATION_DIRECT = "direct_coarse"
LOCALIZATION_FINE_TO_COARSE = "fine_to_coarse"
SCHEMA_VERSION = 1


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    """Resolve absolute paths or paths relative to the repository/manifest."""

    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    for candidate in (REPO_ROOT / path, manifest_path.parent / path):
        if candidate.exists():
            return candidate.resolve()
    return (manifest_path.parent / path).resolve()


class FormalEvaluationError(RuntimeError):
    """Raised when a source-lock or formal-protocol invariant is violated."""


class SplitOutputs:
    def __init__(
        self,
        *,
        volume_ids: list[str],
        case_probabilities: np.ndarray,
        source_max_probabilities: np.ndarray,
        case_targets: np.ndarray,
        patch_targets: np.ndarray,
        localization_scores: np.ndarray,
    ) -> None:
        self.volume_ids = volume_ids
        self.case_probabilities = case_probabilities
        self.source_max_probabilities = source_max_probabilities
        self.case_targets = case_targets
        self.patch_targets = patch_targets
        self.localization_scores = localization_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Completed fold directory containing config.json and best.pt.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to evaluate; defaults to SOURCE_RUN_DIR/best.pt.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--save-localization-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save validation/test localization scores and targets as NPZ.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        help="Smoke-test limiter; any use marks the output non-formal.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        help="Smoke-test limiter; any use marks the output non-formal.",
    )
    return parser.parse_args()


def value_or_default(mapping: dict[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def fine_weight(config: dict[str, Any]) -> float:
    return float(
        value_or_default(
            config.get("innovation", {}),
            "fine_annotation_supervision_weight",
            0.0,
        )
    )


def source_smooth_temperature(config: dict[str, Any]) -> float:
    return float(
        value_or_default(
            config.get("innovation", {}),
            "smooth_or_temperature",
            1.0,
        )
    )


def localization_route(config: dict[str, Any]) -> str:
    return LOCALIZATION_FINE_TO_COARSE if fine_weight(config) > 0.0 else LOCALIZATION_DIRECT


def model_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    innovation = config["innovation"]
    input_processing = config["input_processing"]
    annotation_processing = config["annotation_processing"]
    return {
        "num_classes": int(config["num_output_classes"]),
        "pretrained": False,
        "mil_head": str(value_or_default(innovation, "mil_head", "basic")),
        "class_token_heads": int(value_or_default(innovation, "class_token_heads", 8)),
        "class_token_dropout": float(
            value_or_default(innovation, "class_token_dropout", 0.1)
        ),
        "class_token_decoder_depth": int(
            value_or_default(innovation, "class_token_decoder_depth", 1)
        ),
        "class_token_coord_embedding": bool(
            value_or_default(innovation, "coordinate_embedding", False)
        ),
        "class_token_coord_mode": str(
            value_or_default(innovation, "coordinate_mode", "add")
        ),
        "class_token_anatomical_prior": bool(
            value_or_default(innovation, "anatomical_prior", False)
        ),
        "class_token_prior_gamma": float(
            value_or_default(innovation, "anatomical_prior_gamma", 1.0)
        ),
        "class_token_prior_grid": list(
            value_or_default(innovation, "anatomical_prior_grid", COARSE_GRID)
        ),
        "class_token_residual_init": float(
            value_or_default(innovation, "class_token_residual_init", 0.1)
        ),
        "anatomical_evidence": bool(
            value_or_default(innovation, "crop_aware_anatomical_evidence", False)
        ),
        "anatomical_evidence_hidden_dim": int(
            value_or_default(innovation, "anatomical_evidence_hidden_dim", 64)
        ),
        "anatomical_evidence_gate_init": float(
            value_or_default(innovation, "anatomical_evidence_gate_init", 0.1)
        ),
        "anatomical_pad_shape": list(input_processing["pad_or_crop_shape"]),
        "anatomical_crop_shape": list(input_processing["crop_shape"]),
        "global_local_fusion": bool(
            value_or_default(innovation, "global_local_fusion", False)
        ),
        "local_case_pooling": str(
            value_or_default(innovation, "local_case_pooling", "max")
        ),
        "local_case_topk": int(value_or_default(innovation, "local_case_topk", 4)),
        "adaptive_pool_topks": list(
            value_or_default(innovation, "adaptive_pool_topks", [1, 4, 16, 0])
        ),
        "adaptive_pool_init_weights": list(
            value_or_default(
                innovation,
                "adaptive_pool_init_weights",
                [0.05, 0.85, 0.08, 0.02],
            )
        ),
        "gwrp_decay": float(value_or_default(innovation, "gwrp_decay", 0.996)),
        "fusion_local_init": float(
            value_or_default(innovation, "fusion_local_init", 0.8)
        ),
        "mct_attention_residual_max": float(
            value_or_default(innovation, "mct_attention_residual_max", 0.25)
        ),
        "debug_breakpoints": False,
        "deterministic_adaptive_pool": bool(config["deterministic"]),
        "patch_grid_protocol": str(annotation_processing["patch_grid_protocol"]),
        "patch_token_pooling": str(
            value_or_default(innovation, "patch_token_pooling", "mean")
        ),
        "smooth_or_temperature": source_smooth_temperature(config),
        "fine_annotation_supervision": fine_weight(config) > 0.0,
        "fine_annotation_shape": FINE_GRID,
    }


def architecture_signature(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = model_kwargs_from_config(config)
    return {
        "backbone": config.get("backbone"),
        "num_output_classes": int(config["num_output_classes"]),
        "model_kwargs": kwargs,
        "fine_coarse_consistency_weight": float(
            value_or_default(
                config.get("innovation", {}),
                "fine_coarse_consistency_weight",
                0.0,
            )
        ),
    }


def validate_source_config(config: dict[str, Any]) -> None:
    if config.get("backbone") != "vjepa2_1_b":
        raise FormalEvaluationError("CSEA formal evaluator supports V-JEPA 2.1-B only")
    if not bool(config.get("deterministic")):
        raise FormalEvaluationError("Source run was not recorded as deterministic")
    grid = config.get("annotation_processing", {}).get("patch_grid_protocol")
    if grid != ANATOMICAL_GRID_V2:
        raise FormalEvaluationError(
            f"CSEA formal table requires {ANATOMICAL_GRID_V2}, got {grid!r}"
        )
    innovation = config.get("innovation", {})
    if value_or_default(innovation, "mil_head", "basic") != "basic":
        raise FormalEvaluationError("Formal CSEA table is locked to the basic patch head")
    if bool(value_or_default(innovation, "global_local_fusion", False)):
        raise FormalEvaluationError("Formal CSEA table excludes a separate global/local case head")
    for key in ("case_loss_weight", "local_case_loss_weight", "class_token_case_loss_weight"):
        if float(value_or_default(innovation, key, 0.0)) != 0.0:
            raise FormalEvaluationError(f"Unexpected case-supervision term {key}")
    if len(trainmod.CLASSES) != len(trainmod.SELECTED_IDX):
        raise FormalEvaluationError("Class/index mapping is internally inconsistent")


def audit_source_provenance(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Record source drift without confusing it with prediction drift.

    Baseline/PASE predate the Fine/GAC additions to the shared trainer and
    model files, so their recorded training-source hashes cannot equal the
    current evaluator dependencies.  We preserve both hashes here and rely on
    ``verify_source_max_predictions`` as the strict functional gate: the
    current model implementation must still reproduce every saved source
    max-pooling probability before its CSEA numbers are accepted.
    """

    expected_files = config.get("source_provenance", {}).get("files", {})
    if not expected_files:
        raise FormalEvaluationError("Source config lacks source_provenance.files")
    checked: dict[str, dict[str, Any]] = {}
    for raw_path, expected_hash in expected_files.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Source provenance file is missing: {path}")
        actual_hash = trainmod.sha256_file(path)
        checked[str(path.resolve())] = {
            "training_recorded_sha256": expected_hash,
            "evaluation_time_sha256": actual_hash,
            "matches_training_source": actual_hash == expected_hash,
        }
    return checked


def collect_data_fingerprint_fold_aware(
    manifest_paths: tuple[Path, ...],
    expected_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    """Fingerprint one CV fold without assuming Fold-1 split sizes.

    The fingerprint is fold-aware and portable: identifiers are based on the
    manifest volume ID instead of machine-specific absolute data paths.
    """

    split_names = ("train", "val", "test")
    normalized_counts = {
        split: int(expected_counts[split]) for split in split_names
    }
    if set(expected_counts) != set(split_names):
        raise FormalEvaluationError(
            "Source num_cases must contain exactly train, val, and test"
        )
    if any(value <= 0 for value in normalized_counts.values()):
        raise FormalEvaluationError("Source num_cases values must be positive")

    image_digest = hashlib.sha256()
    annotation_digest = hashlib.sha256()
    annotation_files = 0
    image_files = 0
    all_ids: set[str] = set()
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    required_columns = {
        "volume_id",
        "split",
        "image_path",
        "annotation_dir",
        *(f"{class_name}_label" for class_name in trainmod.CLASSES),
    }
    for split_name, manifest_path in zip(split_names, manifest_paths):
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                missing = sorted(required_columns - set(reader.fieldnames or ()))
                raise FormalEvaluationError(
                    f"{manifest_path} lacks required columns: {missing}"
                )
            rows = list(reader)
        if len(rows) != normalized_counts[split_name]:
            raise FormalEvaluationError(
                f"{split_name} manifest has {len(rows)} rows, "
                f"expected {normalized_counts[split_name]}"
            )
        split_ids = [row["volume_id"] for row in rows]
        if len(set(split_ids)) != len(split_ids):
            raise FormalEvaluationError(
                f"{split_name} manifest contains duplicate volume_id values"
            )
        overlap = all_ids.intersection(split_ids)
        if overlap:
            raise FormalEvaluationError(
                f"Manifest splits overlap at volume IDs: {sorted(overlap)[:5]}"
            )
        all_ids.update(split_ids)
        rows_by_split[split_name] = rows
        for row in rows:
            if row["split"] != split_name:
                raise FormalEvaluationError(
                    f"{manifest_path}: {row['volume_id']} has split={row['split']!r}"
                )
            image_path = resolve_manifest_path(row["image_path"], manifest_path)
            annotation_dir = resolve_manifest_path(
                row["annotation_dir"], manifest_path
            )
            image_identifier = f"{split_name}/{row['volume_id']}/{image_path.name}"
            annotation_identifier = f"{split_name}/{row['volume_id']}"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not annotation_dir.is_dir():
                raise FileNotFoundError(annotation_dir)
            image_stat = image_path.stat()
            image_digest.update(
                (
                    f"{image_identifier}\0{image_stat.st_size}\0"
                    f"{image_stat.st_mtime_ns}\n"
                ).encode("utf-8")
            )
            image_files += 1
            for class_name in trainmod.CLASSES:
                annotation_path = annotation_dir / f"{class_name}.npz"
                positive = float(row[f"{class_name}_label"]) > 0.0
                if positive and not annotation_path.is_file():
                    raise FileNotFoundError(
                        f"Positive label lacks annotation: {annotation_path}"
                    )
                if not annotation_path.is_file():
                    continue
                annotation_relative_file = (
                    f"{annotation_identifier}/{annotation_path.name}"
                )
                annotation_digest.update(
                    (
                        f"{annotation_relative_file}\0"
                        f"{trainmod.sha256_file(annotation_path)}\n"
                    ).encode("utf-8")
                )
                annotation_files += 1
    return (
        {
            "image_files": image_files,
            "image_path_size_mtime_sha256": image_digest.hexdigest(),
            "annotation_files": annotation_files,
            "annotation_content_sha256": annotation_digest.hexdigest(),
            "case_counts": normalized_counts,
            "total_cases": len(all_ids),
        },
        rows_by_split,
    )


def verify_data_provenance(
    source_run_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    manifest_paths = tuple(
        Path(config["split_protocol"][name]).resolve()
        for name in ("train", "val", "test")
    )
    manifest_record = source_run_dir / "manifest_sha256.json"
    fingerprint_record = source_run_dir / "data_fingerprint.json"
    actual_manifest_hashes = {
        str(path): trainmod.sha256_file(path) for path in manifest_paths
    }
    actual_fingerprint, _ = collect_data_fingerprint_fold_aware(
        manifest_paths,
        config["num_cases"],
    )
    # Strict internal runs may carry separate immutable provenance records.
    # Public runs created by the standalone trainer do not require them; when
    # present, however, they are checked exactly.
    if manifest_record.is_file():
        expected_manifest_hashes = json.loads(
            manifest_record.read_text(encoding="utf-8")
        )
        if actual_manifest_hashes != expected_manifest_hashes:
            raise FormalEvaluationError("Manifest SHA-256 values changed after training")
    if fingerprint_record.is_file():
        expected_fingerprint = json.loads(
            fingerprint_record.read_text(encoding="utf-8")
        )
        if actual_fingerprint != expected_fingerprint:
            raise FormalEvaluationError("Dataset metadata fingerprint changed after training")
    return actual_manifest_hashes, actual_fingerprint


def build_model_without_pretrained(config: dict[str, Any]) -> torch.nn.Module:
    spec = trainmod.BACKBONES[config["backbone"]]
    module = trainmod.load_module(
        spec.model_file,
        "patchchestct_vjepa2_1_model_for_formal_csea_evaluation",
    )
    model_cls = getattr(module, spec.class_name)
    return model_cls(**model_kwargs_from_config(config))


def make_loader(
    config: dict[str, Any],
    split_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    inputs = config["input_processing"]
    dataset = trainmod.PatchChestCTPatchDataset(
        Path(config["split_protocol"][split_name]),
        trainmod.CLASSES,
        tuple(int(v) for v in inputs["pad_or_crop_shape"]),
        tuple(int(v) for v in inputs["crop_shape"]),
        tuple(float(v) for v in inputs["hu_clip"]),
        random_crop=False,
        input_mode=str(inputs["vjepa_input_mode"]),
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=trainmod.seed_worker,
        generator=generator,
    )


@torch.no_grad()
def collect_split(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    config: dict[str, Any],
    split_name: str,
    max_batches: int | None,
) -> SplitOutputs:
    model.eval()
    route = localization_route(config)
    grid_protocol = str(config["annotation_processing"]["patch_grid_protocol"])
    source_temperature = source_smooth_temperature(config)

    volume_ids: list[str] = []
    csea_probabilities: list[np.ndarray] = []
    source_max_probabilities: list[np.ndarray] = []
    case_targets_all: list[np.ndarray] = []
    patch_targets_all: list[np.ndarray] = []
    localization_scores_all: list[np.ndarray] = []

    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"{split_name}-raw-logit-csea", leave=False)
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        patch_target24 = batch["patch_target_24"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            raw_outputs = trainmod.forward_patch_model(model, images, batch, device)
        logits, outputs = trainmod.unpack_patch_outputs(raw_outputs)
        selected_coarse_logits = logits[:, trainmod.SELECTED_IDX].float()
        expected_coarse_shape = (
            images.shape[0],
            len(trainmod.CLASSES),
            *COARSE_GRID,
        )
        if tuple(selected_coarse_logits.shape) != expected_coarse_shape:
            raise FormalEvaluationError(
                f"Coarse logits have shape {tuple(selected_coarse_logits.shape)}, "
                f"expected {expected_coarse_shape}"
            )

        csea_logits = smooth_logmeanexp_pool3d(
            selected_coarse_logits,
            output_shape=CSEA_OUTPUT_SHAPE,
            temperature=CSEA_TEMPERATURE,
        )[:, :, 0, 0, 0]
        csea_probs = csea_logits.sigmoid()
        source_max_probs = selected_coarse_logits.sigmoid().amax(dim=(2, 3, 4))

        if route == LOCALIZATION_DIRECT:
            localization_scores = selected_coarse_logits.sigmoid()
        else:
            fine_logits = outputs.get("fine_patch_logits")
            if fine_logits is None:
                raise FormalEvaluationError(
                    "Fine localization route requested but fine_patch_logits are absent"
                )
            selected_fine_logits = fine_logits[:, trainmod.SELECTED_IDX].float()
            expected_fine_shape = (
                images.shape[0],
                len(trainmod.CLASSES),
                *FINE_GRID,
            )
            if tuple(selected_fine_logits.shape) != expected_fine_shape:
                raise FormalEvaluationError(
                    f"Fine logits have shape {tuple(selected_fine_logits.shape)}, "
                    f"expected {expected_fine_shape}"
                )
            localization_scores = smooth_logmeanexp_pool3d(
                selected_fine_logits,
                output_shape=COARSE_GRID,
                temperature=source_temperature,
            ).sigmoid()

        patch_target6 = reduce_patch_target_24_to_6(
            patch_target24,
            protocol=grid_protocol,
        )
        if localization_scores.shape != patch_target6.shape:
            raise FormalEvaluationError(
                f"Localization/target mismatch: {tuple(localization_scores.shape)} "
                f"vs {tuple(patch_target6.shape)}"
            )

        volume_ids.extend(trainmod.get_volume_ids(batch["volume_id"]))
        csea_probabilities.append(csea_probs.detach().cpu().numpy().astype(np.float32))
        source_max_probabilities.append(
            source_max_probs.detach().cpu().numpy().astype(np.float32)
        )
        case_targets_all.append(
            batch["case_target"].detach().cpu().numpy().astype(np.uint8)
        )
        patch_targets_all.append(
            patch_target6.detach().cpu().numpy().astype(np.uint8, copy=False)
        )
        localization_scores_all.append(
            localization_scores.detach().cpu().numpy().astype(np.float32)
        )

    if not csea_probabilities:
        raise FormalEvaluationError(f"No batches were produced for {split_name}")
    return SplitOutputs(
        volume_ids=volume_ids,
        case_probabilities=np.concatenate(csea_probabilities, axis=0),
        source_max_probabilities=np.concatenate(source_max_probabilities, axis=0),
        case_targets=np.concatenate(case_targets_all, axis=0),
        patch_targets=np.concatenate(patch_targets_all, axis=0),
        localization_scores=np.concatenate(localization_scores_all, axis=0),
    )


def make_bundle(split: SplitOutputs) -> trainmod.PredictionBundle:
    patch_scores: list[list[float]] = [[] for _ in trainmod.CLASSES]
    patch_targets: list[list[int]] = [[] for _ in trainmod.CLASSES]
    patch_positive_cases = [0 for _ in trainmod.CLASSES]
    for case_index in range(split.localization_scores.shape[0]):
        for class_index in range(len(trainmod.CLASSES)):
            target = split.patch_targets[case_index, class_index]
            if int(target.sum()) == 0:
                continue
            patch_positive_cases[class_index] += 1
            patch_scores[class_index].extend(
                split.localization_scores[case_index, class_index]
                .reshape(-1)
                .astype(float)
                .tolist()
            )
            patch_targets[class_index].extend(
                target.reshape(-1).astype(int).tolist()
            )
    return trainmod.PredictionBundle(
        loss=math.nan,
        total_loss=math.nan,
        fine_loss=math.nan,
        consistency_loss=math.nan,
        volume_ids=split.volume_ids,
        case_probabilities=split.case_probabilities,
        case_targets=split.case_targets,
        patch_scores=patch_scores,
        patch_targets=patch_targets,
        patch_positive_cases=patch_positive_cases,
    )


def verify_source_max_predictions(
    source_run_dir: Path,
    split_name: str,
    split: SplitOutputs,
) -> dict[str, float | int]:
    path = source_run_dir / f"{split_name}_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(split.volume_ids) * len(trainmod.CLASSES)
    if len(rows) != expected_rows:
        raise FormalEvaluationError(
            f"{path} contains {len(rows)} rows, expected {expected_rows}"
        )
    differences: list[float] = []
    row_index = 0
    for case_index, volume_id in enumerate(split.volume_ids):
        for class_index, class_name in enumerate(trainmod.CLASSES):
            row = rows[row_index]
            row_index += 1
            if row["volume_id"] != volume_id or row["class"] != class_name:
                raise FormalEvaluationError(
                    f"Source prediction order differs at row {row_index}: "
                    f"{row['volume_id']}/{row['class']} vs {volume_id}/{class_name}"
                )
            expected_label = int(split.case_targets[case_index, class_index])
            if int(row["true_label"]) != expected_label:
                raise FormalEvaluationError(
                    f"Source label differs at {volume_id}/{class_name}"
                )
            differences.append(
                abs(
                    float(row["probability"])
                    - float(split.source_max_probabilities[case_index, class_index])
                )
            )
    max_difference = max(differences, default=0.0)
    if max_difference > 5e-7:
        raise FormalEvaluationError(
            f"{split_name} source max-pooling predictions were not reproduced; "
            f"max absolute difference={max_difference:.9g}"
        )
    return {
        "num_predictions": len(differences),
        "max_abs_probability_difference": max_difference,
        "mean_abs_probability_difference": mean(differences) if differences else 0.0,
    }


def select_patch_thresholds(
    val_bundle: trainmod.PredictionBundle,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, float | int | str]] = []
    for class_index, class_name in enumerate(trainmod.CLASSES):
        labels = val_bundle.patch_targets[class_index]
        scores = val_bundle.patch_scores[class_index]
        dsc, threshold, counts = trainmod.best_dice(labels, scores)
        thresholds[class_name] = threshold
        rows.append(
            {
                "class": class_name,
                "threshold": threshold,
                "val_dsc": dsc,
                "val_positive_annotation_cases": val_bundle.patch_positive_cases[class_index],
                "val_positive_cells": sum(labels),
                **{f"val_{key}": value for key, value in counts.items()},
            }
        )
    return thresholds, rows


def dice_at_threshold(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> tuple[float, dict[str, float]]:
    predictions = [int(score > threshold) for score in scores]
    tp = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    fp = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    fn = sum(1 for label, pred in zip(labels, predictions) if label and not pred)
    tn = sum(1 for label, pred in zip(labels, predictions) if not label and not pred)
    dsc = trainmod.safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    return dsc, {"tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


def finite_mean_std_percent(values: list[float]) -> tuple[float, float, str]:
    finite = [float(value) * 100.0 for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan, "NA"
    average = mean(finite)
    std = pstdev(finite)
    return average, std, f"{average:.2f} ± {std:.2f}"


def build_metric_outputs(
    test_rows: list[dict[str, float | int | str]],
    patch_thresholds: dict[str, float],
    patch_threshold_rows: list[dict[str, float | int | str]],
    test_bundle: trainmod.PredictionBundle,
    config: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    patch_threshold_row_by_class = {
        str(row["class"]): row for row in patch_threshold_rows
    }
    per_class_rows: list[dict[str, str]] = []
    completed_patch_threshold_rows: list[dict[str, str]] = []
    val_threshold_dice_values: list[float] = []

    for class_index, row in enumerate(test_rows):
        class_name = str(row["class"])
        threshold = patch_thresholds[class_name]
        dsc_val_threshold, counts = dice_at_threshold(
            test_bundle.patch_targets[class_index],
            test_bundle.patch_scores[class_index],
            threshold,
        )
        val_threshold_dice_values.append(dsc_val_threshold)
        per_class_rows.append(
            {
                "Class": class_name,
                "N": str(row["num_cases"]),
                "Positive Cases": str(row["positive_cases"]),
                "Negative Cases": str(row["negative_cases"]),
                "Case Threshold": f"{float(row['case_threshold']):.8f}",
                "AUROC (%)": f"{float(row['auroc']) * 100.0:.8f}",
                "AUPRC (%)": f"{float(row['auprc']) * 100.0:.8f}",
                "Macro-F1 (%)": f"{float(row['macro_f1']) * 100.0:.8f}",
                "BACC (%)": f"{float(row['bacc']) * 100.0:.8f}",
                "Patch-AUPRC (%)": f"{float(row['patch_auprc']) * 100.0:.8f}",
                "Patch-DSC Oracle (%)": f"{float(row['patch_dsc']) * 100.0:.8f}",
                "Patch Oracle Threshold": f"{float(row['patch_threshold']):.8f}",
                "Patch-DSC@ValThr (%)": f"{dsc_val_threshold * 100.0:.8f}",
                "Patch Val Threshold": f"{threshold:.8f}",
                "Patch Positive Cases": str(row["patch_positive_cases"]),
                "Patch Positive Cells": str(row["patch_positive_cells"]),
                "Patch TP@ValThr": str(int(counts["tp"])),
                "Patch FP@ValThr": str(int(counts["fp"])),
                "Patch FN@ValThr": str(int(counts["fn"])),
                "Patch TN@ValThr": str(int(counts["tn"])),
            }
        )
        val_row = patch_threshold_row_by_class[class_name]
        completed_patch_threshold_rows.append(
            {
                "Class": class_name,
                "Patch Threshold": f"{threshold:.8f}",
                "Selection Split": "validation",
                "Selection Objective": "maximum pooled-cell DSC on positive-annotation cases",
                "Val Patch-DSC (%)": f"{float(val_row['val_dsc']) * 100.0:.8f}",
                "Val Positive Annotation Cases": str(
                    val_row["val_positive_annotation_cases"]
                ),
                "Val Positive Cells": str(val_row["val_positive_cells"]),
                "Test Patch-DSC@ValThr (%)": f"{dsc_val_threshold * 100.0:.8f}",
            }
        )

    metric_map = {
        "AUROC": [float(row["auroc"]) for row in test_rows],
        "AUPRC": [float(row["auprc"]) for row in test_rows],
        "Macro-F1": [float(row["macro_f1"]) for row in test_rows],
        "BACC": [float(row["bacc"]) for row in test_rows],
        "Patch-AUPRC": [float(row["patch_auprc"]) for row in test_rows],
        "Patch-DSC Oracle": [float(row["patch_dsc"]) for row in test_rows],
        "Patch-DSC@ValThr": val_threshold_dice_values,
    }
    summary: dict[str, str] = {
        "Backbone": trainmod.BACKBONES[config["backbone"]].display_name,
        "Case Aggregation": "CSEA raw-logit LogMeanExp",
        "CSEA Temperature": f"{CSEA_TEMPERATURE:.8f}",
        "Localization Route": localization_route(config),
        "Mean/std Scope": "over nine abnormalities",
    }
    for metric_name, values in metric_map.items():
        average, std, formatted = finite_mean_std_percent(values)
        summary[f"{metric_name} (%)"] = formatted
        summary[f"{metric_name} Mean (%)"] = (
            "NA" if not math.isfinite(average) else f"{average:.8f}"
        )
        summary[f"{metric_name} Std (%)"] = (
            "NA" if not math.isfinite(std) else f"{std:.8f}"
        )
    return summary, per_class_rows, completed_patch_threshold_rows


def save_case_predictions(
    path: Path,
    split: SplitOutputs,
    case_thresholds: dict[str, float],
) -> None:
    rows: list[dict[str, str]] = []
    for case_index, volume_id in enumerate(split.volume_ids):
        for class_index, class_name in enumerate(trainmod.CLASSES):
            probability = float(split.case_probabilities[case_index, class_index])
            threshold = case_thresholds[class_name]
            rows.append(
                {
                    "volume_id": volume_id,
                    "class": class_name,
                    "true_label": str(int(split.case_targets[case_index, class_index])),
                    "probability": f"{probability:.8f}",
                    "source_max_probability": (
                        f"{float(split.source_max_probabilities[case_index, class_index]):.8f}"
                    ),
                    "threshold": f"{threshold:.8f}",
                    "predicted_label": str(int(probability >= threshold)),
                    "case_aggregation": "csea_raw_logit_logmeanexp",
                    "csea_temperature": f"{CSEA_TEMPERATURE:.8f}",
                }
            )
    trainmod.write_csv(path, rows)


def save_localization_maps(path: Path, split: SplitOutputs, route: str) -> None:
    np.savez_compressed(
        path,
        volume_ids=np.asarray(split.volume_ids),
        patch_targets=split.patch_targets,
        localization_scores=split.localization_scores,
        localization_route=np.asarray(route),
    )


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    config_path = source_run_dir / "config.json"
    checkpoint_path = (args.checkpoint or source_run_dir / "best.pt").resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing CSEA output directory: {output_dir}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_source_config(config)
    source_hash_audit = audit_source_provenance(config)
    manifest_hashes, data_fingerprint = verify_data_provenance(source_run_dir, config)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FormalEvaluationError("CUDA was requested but is unavailable")
    seed = int(config["seed"])
    trainmod.seed_everything(seed, deterministic=True)
    batch_size = int(args.batch_size or config["batch_size"])
    num_workers = int(
        config["num_workers"] if args.num_workers is None else args.num_workers
    )
    use_amp = bool(config["amp"] and device.type == "cuda")

    model = build_model_without_pretrained(config)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    if checkpoint_epoch != int(config["best_epoch"]):
        raise FormalEvaluationError(
            f"Checkpoint epoch {checkpoint_epoch} != config best_epoch {config['best_epoch']}"
        )
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise FormalEvaluationError("Checkpoint lacks its training config")
    if architecture_signature(checkpoint_config) != architecture_signature(config):
        raise FormalEvaluationError("Checkpoint architecture config differs from source config")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = model.to(device)

    val_loader = make_loader(config, "val", batch_size, num_workers, device)
    test_loader = make_loader(config, "test", batch_size, num_workers, device)
    val_split = collect_split(
        model=model,
        loader=val_loader,
        device=device,
        use_amp=use_amp,
        config=config,
        split_name="val",
        max_batches=args.max_val_batches,
    )
    test_split = collect_split(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=use_amp,
        config=config,
        split_name="test",
        max_batches=args.max_test_batches,
    )

    formal = args.max_val_batches is None and args.max_test_batches is None
    source_prediction_invariance: dict[str, Any] = {}
    if formal:
        source_prediction_invariance = {
            "validation": verify_source_max_predictions(
                source_run_dir, "val", val_split
            ),
            "test": verify_source_max_predictions(source_run_dir, "test", test_split),
        }

    val_bundle = make_bundle(val_split)
    case_thresholds, case_threshold_rows = trainmod.select_thresholds(
        trainmod.CLASSES,
        val_bundle,
        CASE_THRESHOLD_OBJECTIVE,
    )
    patch_thresholds, patch_threshold_rows = select_patch_thresholds(val_bundle)
    test_bundle = make_bundle(test_split)
    test_rows = trainmod.per_class_metrics(
        trainmod.CLASSES,
        test_bundle,
        case_thresholds,
        compute_patch_dice=True,
    )
    summary_row, per_class_rows, completed_patch_threshold_rows = build_metric_outputs(
        test_rows,
        patch_thresholds,
        patch_threshold_rows,
        test_bundle,
        config,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    running_marker = output_dir / "_RUNNING"
    running_marker.write_text(
        datetime.now().astimezone().isoformat(timespec="seconds") + "\n",
        encoding="utf-8",
    )
    trainmod.write_csv(output_dir / "summary_metrics.csv", [summary_row])
    trainmod.write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    trainmod.save_threshold_csv(output_dir / "thresholds.csv", case_threshold_rows)
    trainmod.write_csv(
        output_dir / "patch_thresholds.csv", completed_patch_threshold_rows
    )
    save_case_predictions(
        output_dir / "val_predictions.csv", val_split, case_thresholds
    )
    save_case_predictions(
        output_dir / "test_predictions.csv", test_split, case_thresholds
    )
    if args.save_localization_maps:
        save_localization_maps(
            output_dir / "val_localization_maps.npz",
            val_split,
            localization_route(config),
        )
        save_localization_maps(
            output_dir / "test_localization_maps.npz",
            test_split,
            localization_route(config),
        )

    checkpoint_sha256 = trainmod.sha256_file(checkpoint_path)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": trainmod.sha256_file(Path(__file__).resolve()),
        "command": [str(value) for value in sys.argv],
        "source_run_dir": str(source_run_dir),
        "source_config": str(config_path.resolve()),
        "source_config_sha256": trainmod.sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_epoch": checkpoint_epoch,
        "source_best_epoch": int(config["best_epoch"]),
        "seed": seed,
        "deterministic_source": bool(config["deterministic"]),
        "deterministic_runtime": trainmod.deterministic_runtime_state(),
        "amp": use_amp,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device_requested": str(device),
        "device_resolved": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "csea": {
            "definition": (
                "sigmoid(LogMeanExp over all 6x12x12 selected coarse raw logits)"
            ),
            "temperature": CSEA_TEMPERATURE,
            "temperature_selection": "pre-locked; no evaluator sweep or CLI override",
            "input": "raw float32 coarse patch logits before sigmoid",
            "output_shape": list(CSEA_OUTPUT_SHAPE),
        },
        "case_threshold_protocol": {
            "selection_split": "validation",
            "per_class": True,
            "objective": CASE_THRESHOLD_OBJECTIVE,
            "application_split": "test",
        },
        "localization": {
            "route": localization_route(config),
            "grid_protocol": config["annotation_processing"]["patch_grid_protocol"],
            "output_grid": list(COARSE_GRID),
            "fine_grid": list(FINE_GRID) if fine_weight(config) > 0.0 else None,
            "fine_to_coarse_temperature": (
                source_smooth_temperature(config) if fine_weight(config) > 0.0 else None
            ),
            "csea_changes_localization": False,
            "patch_subset": "positive-annotation cases per class",
            "patch_dsc_oracle": "test thresholds 0..1 step 0.01; official-comparability only",
            "patch_dsc_val_threshold": (
                "per-class threshold selected on validation positive-annotation cases, "
                "then frozen on test"
            ),
        },
        "num_cases": {
            "validation": len(val_split.volume_ids),
            "test": len(test_split.volume_ids),
        },
        "source_prediction_invariance": source_prediction_invariance,
        "source_hash_audit": source_hash_audit,
        "source_hashes_all_match_training": all(
            bool(record["matches_training_source"])
            for record in source_hash_audit.values()
        ),
        "manifest_sha256_verified": manifest_hashes,
        "data_fingerprint_verified": data_fingerprint,
        "outputs": {
            "summary": str(output_dir / "summary_metrics.csv"),
            "per_class": str(output_dir / "per_class_metrics.csv"),
            "case_thresholds": str(output_dir / "thresholds.csv"),
            "patch_thresholds": str(output_dir / "patch_thresholds.csv"),
            "validation_predictions": str(output_dir / "val_predictions.csv"),
            "test_predictions": str(output_dir / "test_predictions.csv"),
            "validation_localization_maps": (
                str(output_dir / "val_localization_maps.npz")
                if args.save_localization_maps
                else None
            ),
            "test_localization_maps": (
                str(output_dir / "test_localization_maps.npz")
                if args.save_localization_maps
                else None
            ),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(trainmod.jsonable(metadata), indent=2),
        encoding="utf-8",
    )
    success_marker = output_dir / ("_SUCCESS" if formal else "_SMOKE_SUCCESS")
    running_marker.replace(success_marker)

    print(json.dumps(summary_row, indent=2), flush=True)
    print(f"Checkpoint SHA-256: {checkpoint_sha256}", flush=True)
    print(f"Saved non-overwriting CSEA evaluation to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
