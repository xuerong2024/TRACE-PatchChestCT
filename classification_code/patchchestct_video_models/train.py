#!/usr/bin/env python3
"""Fine-tune video backbones on PatchChestCT case labels or patch-supervised MIL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np

# This must be configured before the first CUDA BLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification_code.patchchestct_video_models.dataset import (  # noqa: E402
    CLASSES,
    PatchChestCTVideoDataset,
    compute_case_pos_weight,
    compute_patch_pos_weight,
)
from classification_code.patchchestct_video_models.metrics import (  # noqa: E402
    compute_metrics,
    tune_thresholds,
    write_prediction_json,
)
from classification_code.patchchestct_video_models.models import (  # noqa: E402
    MODEL_SPECS,
    ModelSpec,
    build_model,
    get_model_spec,
    trainable_parameter_count,
)


DATA_ROOT = Path("nnunet_data/Bronchidata/PatchChestCT")
SPLIT_ROOT = DATA_ROOT / "manifests/cv5_validtest_seed2026"


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def deterministic_runtime_state() -> dict[str, Any]:
    return {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("dataset.py"),
        Path(__file__).resolve().with_name("models.py"),
        Path(__file__).resolve().with_name("metrics.py"),
        ROOT / "classification_code/deterministic_ops.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = -1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = bce * (1.0 - pt).clamp_min(0.0).pow(self.gamma)
        if self.alpha >= 0.0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


class AsymmetricLossWithLogits(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs_pos = torch.sigmoid(logits.float())
        probs_neg = 1.0 - probs_pos
        if self.clip > 0.0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(probs_pos.clamp(min=self.eps))
        log_neg = torch.log(probs_neg.clamp(min=self.eps))
        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_pos > 0.0 or self.gamma_neg > 0.0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * (1.0 - pt).clamp_min(0.0).pow(gamma)
        return -loss.mean()


def default_patch_target_shape(spec: ModelSpec) -> tuple[int, int, int]:
    if spec.family != "torchhub_vjepa2_1":
        raise ValueError(f"Patch supervision is currently implemented for vjepa2_1_b/V-JEPA 2.1, got {spec.name}")
    tubelet_size = 2
    patch_size = 16
    if spec.frames % tubelet_size != 0:
        raise ValueError(f"V-JEPA 2.1 frames must be divisible by {tubelet_size}, got {spec.frames}")
    if spec.image_size % patch_size != 0:
        raise ValueError(f"V-JEPA 2.1 image_size must be divisible by {patch_size}, got {spec.image_size}")
    return spec.frames // tubelet_size, spec.image_size // patch_size, spec.image_size // patch_size


def pool_patch_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "max":
        return logits.amax(dim=(2, 3, 4))
    if mode == "mean":
        return logits.mean(dim=(2, 3, 4))
    raise ValueError(f"Unsupported patch pooling mode {mode!r}")


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = probs.float()
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(0, 2, 3, 4))
    union = probs.sum(dim=(0, 2, 3, 4)) + targets.sum(dim=(0, 2, 3, 4))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def token_consistency_loss(
    patch_logits: torch.Tensor,
    evidence_attention: torch.Tensor,
    case_targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if tuple(patch_logits.shape) != tuple(evidence_attention.shape):
        raise RuntimeError(
            f"Token consistency expects attention map shape {tuple(patch_logits.shape)}, "
            f"got {tuple(evidence_attention.shape)}"
        )
    positive_classes = case_targets.float()
    if int((positive_classes > 0).sum().item()) == 0:
        return patch_logits.sum() * 0.0
    patch_dist = F.softmax(patch_logits.flatten(2).float(), dim=-1)
    attention_dist = evidence_attention.flatten(2).float().clamp_min(0.0)
    attention_dist = attention_dist / attention_dist.sum(dim=-1, keepdim=True).clamp_min(eps)
    per_class = F.mse_loss(patch_dist, attention_dist, reduction="none").mean(dim=-1)
    return (per_class * positive_classes).sum() / positive_classes.sum().clamp_min(eps)


def evidence_alignment_loss(
    evidence_attention: torch.Tensor,
    patch_targets: torch.Tensor,
    loss_type: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    if tuple(evidence_attention.shape) != tuple(patch_targets.shape):
        raise RuntimeError(
            f"Evidence alignment expects attention shape {tuple(patch_targets.shape)}, "
            f"got {tuple(evidence_attention.shape)}"
        )
    attention = evidence_attention.flatten(2).float().clamp_min(0.0)
    targets = patch_targets.flatten(2).float().clamp(0.0, 1.0)
    target_sum = targets.sum(dim=-1)
    positive = target_sum > 0.0
    if int(positive.sum().item()) == 0:
        return evidence_attention.sum() * 0.0
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(eps)
    if loss_type == "kl":
        target_dist = targets / target_sum.unsqueeze(-1).clamp_min(eps)
        per_class = (
            target_dist
            * (target_dist.clamp_min(eps).log() - attention.clamp_min(eps).log())
        ).sum(dim=-1)
    elif loss_type == "bce":
        per_class = F.binary_cross_entropy(
            attention.clamp(min=eps, max=1.0 - eps),
            targets,
            reduction="none",
        ).mean(dim=-1)
    else:
        raise ValueError(f"Unsupported evidence alignment loss {loss_type!r}")
    return per_class[positive].mean()


def prediction_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    loss_type: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    if tuple(logits_a.shape) != tuple(logits_b.shape):
        raise RuntimeError(
            f"Prediction consistency expects matching shapes, got {tuple(logits_a.shape)} "
            f"and {tuple(logits_b.shape)}"
        )
    probs_a = torch.sigmoid(logits_a.float()).clamp(min=eps, max=1.0 - eps)
    probs_b = torch.sigmoid(logits_b.float()).clamp(min=eps, max=1.0 - eps)
    if loss_type == "mse":
        return F.mse_loss(probs_a, probs_b)
    if loss_type == "kl":
        kl_ab = probs_a * (probs_a / probs_b).log() + (1.0 - probs_a) * (
            (1.0 - probs_a) / (1.0 - probs_b)
        ).log()
        kl_ba = probs_b * (probs_b / probs_a).log() + (1.0 - probs_b) * (
            (1.0 - probs_b) / (1.0 - probs_a)
        ).log()
        return 0.5 * (kl_ab + kl_ba).mean()
    raise ValueError(f"Unsupported consistency loss {loss_type!r}")


def build_criterion(args: argparse.Namespace, pos_weight: torch.Tensor | None) -> nn.Module:
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.loss == "focal":
        return FocalLossWithLogits(gamma=args.focal_gamma, alpha=args.focal_alpha)
    if args.loss == "asl":
        return AsymmetricLossWithLogits(
            gamma_pos=args.asl_gamma_pos,
            gamma_neg=args.asl_gamma_neg,
            clip=args.asl_clip,
            eps=args.asl_eps,
        )
    raise ValueError(f"Unsupported loss {args.loss!r}")


def make_dataset(
    csv_path: Path,
    spec: ModelSpec,
    classes: list[str],
    args: argparse.Namespace,
    *,
    train: bool,
    max_cases: int | None,
) -> PatchChestCTVideoDataset:
    return PatchChestCTVideoDataset(
        csv_path,
        classes=classes,
        frames=spec.frames,
        image_size=spec.image_size,
        clip_hu=tuple(args.clip_hu),
        mean=spec.mean,
        std=spec.std,
        train=train,
        random_horizontal_flip=args.random_horizontal_flip if train else 0.0,
        allow_unverified_nifti=args.allow_unverified_nifti,
        max_cases=max_cases,
        input_space=args.input_space,
        load_patch_targets=args.supervision == "patch",
        patch_target_shape=args.patch_target_shape if args.supervision == "patch" else None,
        patch_target_reduction=args.patch_target_reduction,
    )


def make_loader(
    dataset: PatchChestCTVideoDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    case_criterion: nn.Module,
    device: torch.device,
    classes: list[str],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    threshold: float = 0.5,
    supervision: str = "case",
    patch_criterion: nn.Module | None = None,
    patch_pooling: str = "max",
    patch_loss_weight: float = 1.0,
    patch_case_loss_weight: float = 1.0,
    patch_dice_weight: float = 0.0,
    token_consistency_weight: float = 0.0,
    evidence_alignment_weight: float = 0.0,
    evidence_alignment_loss_type: str = "kl",
    slice_order_consistency_weight: float = 0.0,
    patch_slice_order_consistency_weight: float = 0.0,
    slice_order_consistency_loss_type: str = "mse",
    slice_order_consistency_backprop_reversed: bool = False,
    progress_desc: str = "",
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_volume_ids: list[str] = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=progress_desc, leave=True, dynamic_ncols=True, file=sys.stdout)
    for step, batch in enumerate(progress, start=1):
        videos = batch["video"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if supervision == "case":
                    logits = model(videos)
                    if not isinstance(logits, torch.Tensor):
                        raise RuntimeError("Case supervision expected tensor logits from the model")
                    loss = case_criterion(logits.float(), targets.float())
                    case_logits_for_metrics = logits
                elif supervision == "patch":
                    if patch_criterion is None:
                        raise RuntimeError("Patch supervision requires patch_criterion")
                    if "patch_target" not in batch:
                        raise RuntimeError("Patch supervision requires patch_target in the dataset")
                    use_slice_order_consistency = is_train and (
                        slice_order_consistency_weight > 0.0 or patch_slice_order_consistency_weight > 0.0
                    )
                    reversed_patch_logits_for_consistency: torch.Tensor | None = None
                    if use_slice_order_consistency and not slice_order_consistency_backprop_reversed:
                        with torch.no_grad():
                            reversed_outputs = model(torch.flip(videos, dims=(2,)), return_patch_logits=True)
                        if not isinstance(reversed_outputs, dict) or "patch_logits" not in reversed_outputs:
                            raise RuntimeError("Slice-order consistency requires reversed patch logits")
                        reversed_patch_logits_for_consistency = reversed_outputs["patch_logits"].detach()
                    outputs = model(videos, return_patch_logits=True)
                    if not isinstance(outputs, dict) or "patch_logits" not in outputs or "logits" not in outputs:
                        raise RuntimeError("Patch supervision requires model(..., return_patch_logits=True)")
                    patch_logits = outputs["patch_logits"]
                    patch_targets = batch["patch_target"].to(device, non_blocking=True)
                    if tuple(patch_logits.shape) != tuple(patch_targets.shape):
                        raise RuntimeError(
                            f"Patch logits shape {tuple(patch_logits.shape)} does not match "
                            f"targets {tuple(patch_targets.shape)}"
                        )
                    patch_loss = patch_criterion(patch_logits.float(), patch_targets.float())
                    if patch_dice_weight > 0.0:
                        patch_loss = patch_loss + patch_dice_weight * dice_loss_from_probs(
                            patch_logits.sigmoid(),
                            patch_targets,
                        )
                    case_logits_for_metrics = pool_patch_logits(patch_logits, patch_pooling)
                    case_loss = case_criterion(case_logits_for_metrics.float(), targets.float())
                    loss = patch_loss_weight * patch_loss + patch_case_loss_weight * case_loss
                    if token_consistency_weight > 0.0:
                        evidence_attention = outputs.get("evidence_attention")
                        if evidence_attention is None:
                            raise RuntimeError(
                                "--token-consistency-weight requires --mil-head class-token "
                                "because only that head returns evidence_attention"
                            )
                        loss = loss + token_consistency_weight * token_consistency_loss(
                            patch_logits,
                            evidence_attention,
                            targets,
                        )
                    if evidence_alignment_weight > 0.0:
                        evidence_attention = outputs.get("evidence_attention")
                        if evidence_attention is None:
                            raise RuntimeError(
                                "--evidence-alignment-weight requires --mil-head class-token "
                                "because only that head returns evidence_attention"
                            )
                        loss = loss + evidence_alignment_weight * evidence_alignment_loss(
                            evidence_attention,
                            patch_targets,
                            evidence_alignment_loss_type,
                        )
                    if use_slice_order_consistency:
                        if slice_order_consistency_backprop_reversed:
                            reversed_outputs = model(torch.flip(videos, dims=(2,)), return_patch_logits=True)
                            if not isinstance(reversed_outputs, dict) or "patch_logits" not in reversed_outputs:
                                raise RuntimeError("Slice-order consistency requires reversed patch logits")
                            reversed_patch_logits = reversed_outputs["patch_logits"]
                        else:
                            if reversed_patch_logits_for_consistency is None:
                                raise RuntimeError("Missing detached reversed patch logits")
                            reversed_patch_logits = reversed_patch_logits_for_consistency
                        if tuple(reversed_patch_logits.shape) != tuple(patch_logits.shape):
                            raise RuntimeError(
                                f"Reversed patch logits shape {tuple(reversed_patch_logits.shape)} does not match "
                                f"forward patch logits {tuple(patch_logits.shape)}"
                            )
                        if slice_order_consistency_weight > 0.0:
                            reversed_case_logits = pool_patch_logits(reversed_patch_logits, patch_pooling)
                            loss = loss + slice_order_consistency_weight * prediction_consistency_loss(
                                case_logits_for_metrics,
                                reversed_case_logits,
                                slice_order_consistency_loss_type,
                            )
                        if patch_slice_order_consistency_weight > 0.0:
                            aligned_reversed_patch_logits = torch.flip(reversed_patch_logits, dims=(2,))
                            loss = loss + patch_slice_order_consistency_weight * prediction_consistency_loss(
                                patch_logits,
                                aligned_reversed_patch_logits,
                                slice_order_consistency_loss_type,
                            )
                else:
                    raise ValueError(f"Unsupported supervision mode {supervision!r}")

            if is_train:
                scaled_loss = loss / max(gradient_accumulation_steps, 1)
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = step % gradient_accumulation_steps == 0 or step == len(loader)
                if should_step:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        probs = torch.sigmoid(case_logits_for_metrics).detach().cpu().float().numpy()
        all_probs.append(probs)
        all_targets.append(targets.detach().cpu().float().numpy())
        all_volume_ids.extend(str(v) for v in batch["volume_id"])
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
        progress.set_postfix(loss=f"{total_loss / max(total_batches, 1):.4f}")

    probs_np = np.concatenate(all_probs, axis=0)
    targets_np = np.concatenate(all_targets, axis=0).astype(np.int32)
    metrics = compute_metrics(targets_np, probs_np, classes, threshold)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return {
        "loss": metrics["loss"],
        "metrics": metrics,
        "probs": probs_np,
        "targets": targets_np,
        "volume_ids": all_volume_ids,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    args: argparse.Namespace,
    spec: ModelSpec,
    classes: list[str],
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": jsonable_args(args),
            "model_spec": spec.to_dict(),
            "classes": classes,
            "metrics": metrics or {},
        },
        path,
    )


def flatten_history_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {"epoch": record["epoch"], "lr": record["lr"]}
    for split in ("train", "val"):
        values = record[split]
        flat[f"{split}_loss"] = values["loss"]
        flat[f"{split}_macro_auprc_ap"] = values["macro_auprc_ap"]
        flat[f"{split}_macro_auroc"] = values["macro_auroc"]
        flat[f"{split}_macro_f1"] = values["macro_f1"]
        flat[f"{split}_macro_balanced_accuracy"] = values["macro_balanced_accuracy"]
        flat[f"{split}_micro_accuracy"] = values["micro_accuracy"]
        flat[f"{split}_micro_f1"] = values["micro_f1"]
        flat[f"{split}_micro_precision"] = values["micro_precision"]
        flat[f"{split}_micro_recall_sensitivity"] = values["micro_recall_sensitivity"]
        flat[f"{split}_micro_specificity"] = values["micro_specificity"]
    return flat


def write_history_files(output_dir: Path, history: list[dict[str, Any]]) -> None:
    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    if not history:
        return
    rows = [flatten_history_record(record) for record in history]
    with (output_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_for_selection(result: dict[str, Any], best_metric: str) -> float:
    if best_metric == "val_loss":
        return -float(result["loss"])
    return float(result["metrics"][best_metric])


def score_history_record(record: dict[str, Any], best_metric: str) -> float:
    if best_metric == "val_loss":
        return -float(record["val"]["loss"])
    return float(record["val"][best_metric])


def best_from_history(history: list[dict[str, Any]], best_metric: str) -> tuple[float, int]:
    best_score = -float("inf")
    best_epoch = 0
    for record in history:
        score = score_history_record(record, best_metric)
        if score > best_score:
            best_score = score
            best_epoch = int(record["epoch"])
    return best_score, best_epoch


def load_existing_history(output_dir: Path, max_epoch: int) -> list[dict[str, Any]]:
    path = output_dir / "history.json"
    if not path.exists():
        return []
    with path.open() as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise RuntimeError(f"{path} did not contain a history list")
    return [record for record in history if int(record.get("epoch", 0)) <= max_epoch]


def sync_cosine_scheduler_to_epoch(
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
) -> None:
    if completed_epochs <= 0:
        return
    t_max = max(int(scheduler.T_max), 1)
    eta_min = float(scheduler.eta_min)
    lrs = [
        eta_min + (float(base_lr) - eta_min) * (1.0 + math.cos(math.pi * completed_epochs / t_max)) / 2.0
        for base_lr in scheduler.base_lrs
    ]
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = lr
    scheduler.last_epoch = completed_epochs
    scheduler._last_lr = lrs


def default_lr(spec: ModelSpec) -> float:
    if spec.family.startswith("hf_"):
        return 1e-5 if spec.family == "hf_vjepa2" else 2e-5
    if spec.family == "torchhub_vjepa2_1":
        return 1e-5
    if spec.family.startswith("torchvision_mvit"):
        return 5e-5
    return 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--hf-model-id", help="Override the registry Hugging Face model id.")
    parser.add_argument("--train-csv", type=Path, default=SPLIT_ROOT / "fold_0/train.csv")
    parser.add_argument("--val-csv", type=Path, default=SPLIT_ROOT / "fold_0/val.csv")
    parser.add_argument("--test-csv", type=Path, default=SPLIT_ROOT / "test.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume model weights and epoch from this checkpoint.")
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument("--frames", type=int, help="Override the registry clip length.")
    parser.add_argument("--image-size", type=int, help="Override the registry spatial crop size.")
    parser.add_argument("--clip-hu", nargs=2, type=float, default=[-1000.0, 1000.0], metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--input-space",
        choices=("display", "direct-npz"),
        default="display",
        help=(
            "display applies the official verify_alignment.py display transform/crop; "
            "direct-npz reads arr_0 full FOV directly and is for debugging only."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa", help="HF attention backend; set empty string to disable.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold-grid-steps", type=int, default=181)
    parser.add_argument("--supervision", choices=("case", "patch"), default="case")
    parser.add_argument("--patch-target-shape", nargs=3, type=int, metavar=("D", "H", "W"))
    parser.add_argument(
        "--patch-target-reduction",
        choices=("adaptive-max", "official-24-to-6"),
        default="adaptive-max",
        help="How manual 24x12x12 labels are reduced when patch-target-shape differs.",
    )
    parser.add_argument("--patch-pooling", choices=("max", "mean"), default="max", help="MIL pooling from patch logits to case logits.")
    parser.add_argument("--patch-loss-weight", type=float, default=1.0, help="Weight for patch-level supervised loss.")
    parser.add_argument("--patch-case-loss-weight", type=float, default=1.0, help="Weight for case-level MIL loss from pooled patch logits.")
    parser.add_argument("--patch-dice-weight", type=float, default=0.0, help="Optional Dice term added to patch-level BCE/focal/ASL.")
    parser.add_argument("--mil-head", choices=("basic", "class-token", "prototype"), default="basic")
    parser.add_argument("--class-token-heads", type=int, default=8)
    parser.add_argument("--class-token-dropout", type=float, default=0.1)
    parser.add_argument("--class-token-coord-embedding", action="store_true")
    parser.add_argument("--class-token-coord-mode", choices=("add", "concat"), default="add")
    parser.add_argument("--class-token-anatomical-prior", action="store_true")
    parser.add_argument("--class-token-prior-gamma", type=float, default=1.0)
    parser.add_argument("--class-token-prior-grid", nargs=3, type=int, default=None, metavar=("D", "H", "W"))
    parser.add_argument("--token-consistency-weight", type=float, default=0.0)
    parser.add_argument("--evidence-alignment-weight", type=float, default=0.0)
    parser.add_argument("--evidence-alignment-loss", choices=("kl", "bce"), default="kl")
    parser.add_argument("--slice-order-consistency-weight", type=float, default=0.0)
    parser.add_argument("--patch-slice-order-consistency-weight", type=float, default=0.0)
    parser.add_argument("--slice-order-consistency-loss", choices=("mse", "kl"), default="mse")
    parser.add_argument(
        "--slice-order-consistency-backprop-reversed",
        action="store_true",
        help="Backprop through the reversed CT branch. This is much more memory hungry.",
    )
    parser.add_argument("--prototype-count", type=int, default=4)
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument("--pos-weight", choices=("auto", "none"), default="auto")
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--loss", choices=("bce", "focal", "asl"), default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=-1.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-eps", type=float, default=1e-8)
    parser.add_argument("--random-horizontal-flip", type=float, default=0.5)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict deterministic kernels (default: enabled).",
    )
    parser.add_argument("--allow-unverified-nifti", action="store_true")
    parser.add_argument("--best-metric", choices=("val_loss", "macro_auprc_ap", "macro_f1"), default="macro_auprc_ap")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed, args.deterministic)
    runtime_determinism = deterministic_runtime_state()
    print("Deterministic runtime:", json.dumps(runtime_determinism, indent=2), flush=True)
    classes = list(args.classes)
    spec = get_model_spec(args.model, hf_model_id=args.hf_model_id, frames=args.frames, image_size=args.image_size)
    if args.supervision == "patch":
        if args.patch_target_shape is None:
            args.patch_target_shape = list(default_patch_target_shape(spec))
        else:
            args.patch_target_shape = [int(v) for v in args.patch_target_shape]
        if len(args.patch_target_shape) != 3:
            raise ValueError("--patch-target-shape must contain D H W")
        if args.patch_loss_weight <= 0.0:
            raise ValueError("--patch-loss-weight must be > 0 for patch supervision")
        if args.token_consistency_weight > 0.0 and args.mil_head != "class-token":
            raise ValueError("--token-consistency-weight is only valid with --mil-head class-token")
        if (
            args.class_token_coord_embedding
            or args.class_token_anatomical_prior
            or args.evidence_alignment_weight > 0.0
        ) and args.mil_head != "class-token":
            raise ValueError("Class-token coordinate/prior/evidence options require --mil-head class-token")
        if args.evidence_alignment_weight < 0.0:
            raise ValueError("--evidence-alignment-weight must be >= 0")
        if args.slice_order_consistency_weight < 0.0 or args.patch_slice_order_consistency_weight < 0.0:
            raise ValueError("Slice-order consistency weights must be >= 0")
        if (
            args.slice_order_consistency_weight > 0.0 or args.patch_slice_order_consistency_weight > 0.0
        ) and args.mil_head != "class-token":
            raise ValueError("Slice-order consistency is currently wired for --mil-head class-token")
    else:
        args.patch_target_shape = None
    if args.class_token_prior_grid is None:
        args.class_token_prior_grid = list(args.patch_target_shape or [32, 24, 24])
    else:
        args.class_token_prior_grid = [int(v) for v in args.class_token_prior_grid]
    if args.lr is None:
        args.lr = default_lr(spec)
    if args.output_dir is None:
        args.output_dir = DATA_ROOT / "video_model_runs" / f"cv5_validtest_seed2026_fold0_{args.model}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    print("Model spec:", json.dumps(spec.to_dict(), indent=2), flush=True)
    print("Output dir:", args.output_dir, flush=True)
    print("Supervision:", args.supervision, "Input space:", args.input_space, flush=True)
    if args.supervision == "patch":
        print("Patch MIL head:", args.mil_head, flush=True)
        if args.mil_head == "class-token":
            print(
                "Class-token anatomy options:",
                {
                    "coord_embedding": args.class_token_coord_embedding,
                    "coord_mode": args.class_token_coord_mode,
                    "anatomical_prior": args.class_token_anatomical_prior,
                    "prior_gamma": args.class_token_prior_gamma,
                    "prior_grid": args.class_token_prior_grid,
                    "evidence_alignment_weight": args.evidence_alignment_weight,
                    "evidence_alignment_loss": args.evidence_alignment_loss,
                },
                flush=True,
            )
        if args.slice_order_consistency_weight > 0.0 or args.patch_slice_order_consistency_weight > 0.0:
            print(
                "Slice-order consistency:",
                {
                    "case_weight": args.slice_order_consistency_weight,
                    "patch_weight": args.patch_slice_order_consistency_weight,
                    "loss": args.slice_order_consistency_loss,
                    "backprop_reversed": args.slice_order_consistency_backprop_reversed,
                },
                flush=True,
            )
    if args.patch_target_shape is not None:
        print("Patch target shape:", args.patch_target_shape, flush=True)
    print("Device:", device, flush=True)
    train_set = make_dataset(args.train_csv, spec, classes, args, train=True, max_cases=args.max_train_cases)
    val_set = make_dataset(args.val_csv, spec, classes, args, train=False, max_cases=args.max_val_cases)
    train_loader = make_loader(train_set, args.batch_size, args.num_workers, device, shuffle=True, generator=generator)
    val_loader = make_loader(val_set, args.batch_size, args.num_workers, device, shuffle=False, generator=generator)

    model = build_model(
        spec,
        num_classes=len(classes),
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        attn_implementation=args.attn_implementation or None,
        patch_supervision=args.supervision == "patch",
        mil_head=args.mil_head,
        class_token_heads=args.class_token_heads,
        class_token_dropout=args.class_token_dropout,
        class_token_coord_embedding=args.class_token_coord_embedding,
        class_token_coord_mode=args.class_token_coord_mode,
        class_token_anatomical_prior=args.class_token_anatomical_prior,
        class_token_prior_gamma=args.class_token_prior_gamma,
        class_token_prior_grid=args.class_token_prior_grid,
        patch_output_shape=args.patch_target_shape,
        prototype_count=args.prototype_count,
        prototype_temperature=args.prototype_temperature,
        deterministic=args.deterministic,
    ).to(device)
    total_params, trainable_params = trainable_parameter_count(model)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}", flush=True)
    deterministic_replacements = getattr(model, "_deterministic_replacements", None)
    if deterministic_replacements is not None:
        print("Deterministic replacements:", json.dumps(deterministic_replacements, indent=2), flush=True)

    case_pos_weight = None
    patch_pos_weight = None
    if args.pos_weight == "auto":
        case_pos_weight = compute_case_pos_weight(train_set, args.max_pos_weight).to(device)
        print("Using case pos_weight:", {c: float(v) for c, v in zip(classes, case_pos_weight.detach().cpu())}, flush=True)
        if args.supervision == "patch":
            patch_pos_weight = compute_patch_pos_weight(train_set, args.max_pos_weight).to(device).view(1, -1, 1, 1, 1)
            print(
                "Using patch pos_weight:",
                {c: float(v) for c, v in zip(classes, patch_pos_weight.flatten().detach().cpu())},
                flush=True,
            )
    case_criterion = build_criterion(args, case_pos_weight)
    patch_criterion = build_criterion(args, patch_pos_weight) if args.supervision == "patch" else None
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0))
        history = load_existing_history(args.output_dir, start_epoch)
        if history:
            best_score, best_epoch = best_from_history(history, args.best_metric)
        sync_cosine_scheduler_to_epoch(scheduler, optimizer, start_epoch)
        print(
            f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}; "
            f"loaded {len(history)} history rows; next epoch is {start_epoch + 1}.",
            flush=True,
        )

    config = {
        "args": jsonable_args(args),
        "model_spec": spec.to_dict(),
        "classes": classes,
        "num_train_cases": len(train_set),
        "num_val_cases": len(val_set),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "deterministic_replacements": deterministic_replacements,
        "deterministic_runtime": runtime_determinism,
        "source_sha256": source_provenance(),
    }
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} - train", flush=True)
        train_result = run_epoch(
            model,
            train_loader,
            case_criterion,
            device,
            classes,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            threshold=args.threshold,
            supervision=args.supervision,
            patch_criterion=patch_criterion,
            patch_pooling=args.patch_pooling,
            patch_loss_weight=args.patch_loss_weight,
            patch_case_loss_weight=args.patch_case_loss_weight,
            patch_dice_weight=args.patch_dice_weight,
            token_consistency_weight=args.token_consistency_weight,
            evidence_alignment_weight=args.evidence_alignment_weight,
            evidence_alignment_loss_type=args.evidence_alignment_loss,
            slice_order_consistency_weight=args.slice_order_consistency_weight,
            patch_slice_order_consistency_weight=args.patch_slice_order_consistency_weight,
            slice_order_consistency_loss_type=args.slice_order_consistency_loss,
            slice_order_consistency_backprop_reversed=args.slice_order_consistency_backprop_reversed,
            progress_desc=f"train e{epoch}/{args.epochs}",
        )
        print(f"Epoch {epoch}/{args.epochs} - val", flush=True)
        with torch.no_grad():
            val_result = run_epoch(
                model,
                val_loader,
                case_criterion,
                device,
                classes,
                use_amp=use_amp,
                threshold=args.threshold,
                supervision=args.supervision,
                patch_criterion=patch_criterion,
                patch_pooling=args.patch_pooling,
                patch_loss_weight=args.patch_loss_weight,
                patch_case_loss_weight=args.patch_case_loss_weight,
                patch_dice_weight=args.patch_dice_weight,
                token_consistency_weight=args.token_consistency_weight,
                evidence_alignment_weight=args.evidence_alignment_weight,
                evidence_alignment_loss_type=args.evidence_alignment_loss,
                slice_order_consistency_weight=args.slice_order_consistency_weight,
                patch_slice_order_consistency_weight=args.patch_slice_order_consistency_weight,
                slice_order_consistency_loss_type=args.slice_order_consistency_loss,
                slice_order_consistency_backprop_reversed=args.slice_order_consistency_backprop_reversed,
                progress_desc=f"val e{epoch}/{args.epochs}",
            )
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": {
                "loss": train_result["loss"],
                "macro_auprc_ap": train_result["metrics"]["macro_auprc_ap"],
                "macro_auroc": train_result["metrics"]["macro_auroc"],
                "macro_f1": train_result["metrics"]["macro_f1"],
                "macro_balanced_accuracy": train_result["metrics"]["macro_balanced_accuracy"],
                "micro_accuracy": train_result["metrics"]["micro_accuracy"],
                "micro_f1": train_result["metrics"]["micro_f1"],
                "micro_precision": train_result["metrics"]["micro_precision"],
                "micro_recall_sensitivity": train_result["metrics"]["micro_recall_sensitivity"],
                "micro_specificity": train_result["metrics"]["micro_specificity"],
            },
            "val": {
                "loss": val_result["loss"],
                "macro_auprc_ap": val_result["metrics"]["macro_auprc_ap"],
                "macro_auroc": val_result["metrics"]["macro_auroc"],
                "macro_f1": val_result["metrics"]["macro_f1"],
                "macro_balanced_accuracy": val_result["metrics"]["macro_balanced_accuracy"],
                "micro_accuracy": val_result["metrics"]["micro_accuracy"],
                "micro_f1": val_result["metrics"]["micro_f1"],
                "micro_precision": val_result["metrics"]["micro_precision"],
                "micro_recall_sensitivity": val_result["metrics"]["micro_recall_sensitivity"],
                "micro_specificity": val_result["metrics"]["micro_specificity"],
            },
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        save_checkpoint(args.output_dir / "last.pt", model, epoch, args, spec, classes, val_result["metrics"])

        score = metric_for_selection(val_result, args.best_metric)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(args.output_dir / "best.pt", model, epoch, args, spec, classes, val_result["metrics"])

        write_history_files(args.output_dir, history)

    if args.epochs == 0:
        save_checkpoint(args.output_dir / "last.pt", model, 0, args, spec, classes)
        save_checkpoint(args.output_dir / "best.pt", model, 0, args, spec, classes)

    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()

    with torch.no_grad():
        val_result = run_epoch(
            model,
            val_loader,
            case_criterion,
            device,
            classes,
            use_amp=use_amp,
            threshold=args.threshold,
            supervision=args.supervision,
            patch_criterion=patch_criterion,
            patch_pooling=args.patch_pooling,
            patch_loss_weight=args.patch_loss_weight,
            patch_case_loss_weight=args.patch_case_loss_weight,
            patch_dice_weight=args.patch_dice_weight,
            token_consistency_weight=args.token_consistency_weight,
            evidence_alignment_weight=args.evidence_alignment_weight,
            evidence_alignment_loss_type=args.evidence_alignment_loss,
            slice_order_consistency_weight=args.slice_order_consistency_weight,
            patch_slice_order_consistency_weight=args.patch_slice_order_consistency_weight,
            slice_order_consistency_loss_type=args.slice_order_consistency_loss,
            slice_order_consistency_backprop_reversed=args.slice_order_consistency_backprop_reversed,
            progress_desc="final val",
        )
    if args.tune_thresholds:
        thresholds = tune_thresholds(
            val_result["targets"],
            val_result["probs"],
            classes,
            steps=args.threshold_grid_steps,
        )
    else:
        thresholds = {class_name: args.threshold for class_name in classes}

    with (args.output_dir / "val_thresholds.json").open("w") as f:
        json.dump(thresholds, f, indent=2)
    write_prediction_json(
        args.output_dir / "val_predictions.json",
        model_name=args.model,
        checkpoint=str(args.output_dir / "best.pt"),
        csv_path=str(args.val_csv),
        volume_ids=val_result["volume_ids"],
        labels=val_result["targets"],
        probs=val_result["probs"],
        classes=classes,
        thresholds=thresholds,
        extra_summary={"best_epoch": best_epoch, "best_metric": args.best_metric, "best_score": best_score},
    )

    if args.test_csv:
        test_set = make_dataset(args.test_csv, spec, classes, args, train=False, max_cases=args.max_test_cases)
        test_loader = make_loader(test_set, args.batch_size, args.num_workers, device, shuffle=False, generator=generator)
        with torch.no_grad():
            test_result = run_epoch(
                model,
                test_loader,
                case_criterion,
                device,
                classes,
                use_amp=use_amp,
                threshold=args.threshold,
                supervision=args.supervision,
                patch_criterion=patch_criterion,
                patch_pooling=args.patch_pooling,
                patch_loss_weight=args.patch_loss_weight,
                patch_case_loss_weight=args.patch_case_loss_weight,
                patch_dice_weight=args.patch_dice_weight,
                token_consistency_weight=args.token_consistency_weight,
                evidence_alignment_weight=args.evidence_alignment_weight,
                evidence_alignment_loss_type=args.evidence_alignment_loss,
                slice_order_consistency_weight=args.slice_order_consistency_weight,
                patch_slice_order_consistency_weight=args.patch_slice_order_consistency_weight,
                slice_order_consistency_loss_type=args.slice_order_consistency_loss,
                slice_order_consistency_backprop_reversed=args.slice_order_consistency_backprop_reversed,
                progress_desc="test",
            )
        payload = write_prediction_json(
            args.output_dir / "test_predictions.json",
            model_name=args.model,
            checkpoint=str(args.output_dir / "best.pt"),
            csv_path=str(args.test_csv),
            volume_ids=test_result["volume_ids"],
            labels=test_result["targets"],
            probs=test_result["probs"],
            classes=classes,
            thresholds=thresholds,
            extra_summary={"best_epoch": best_epoch, "best_metric": args.best_metric, "best_score": best_score},
        )
        with (args.output_dir / "test_metrics.json").open("w") as f:
            json.dump(payload["summary"]["metrics"], f, indent=2)
        print("Test metrics:", json.dumps(payload["summary"]["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
