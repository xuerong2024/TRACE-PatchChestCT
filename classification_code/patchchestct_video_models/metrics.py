"""Metrics and prediction serialization for PatchChestCT video classifiers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import numpy as np


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan

    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    n_pos = sum(labels)
    if n_pos == 0:
        return math.nan

    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for _, label in pairs[i:j]:
            if label:
                tp += 1
            else:
                fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def confusion_metrics(labels: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(bool)
    preds = probs >= thresholds.reshape(1, -1)

    tp = np.logical_and(labels, preds).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~labels, preds).sum(axis=0).astype(np.float64)
    fn = np.logical_and(labels, ~preds).sum(axis=0).astype(np.float64)
    tn = np.logical_and(~labels, ~preds).sum(axis=0).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    balanced_accuracy = (recall + specificity) / 2.0

    micro_tp = float(tp.sum())
    micro_fp = float(fp.sum())
    micro_fn = float(fn.sum())
    micro_tn = float(tn.sum())
    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2.0 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "tp": tp.astype(int).tolist(),
        "fp": fp.astype(int).tolist(),
        "fn": fn.astype(int).tolist(),
        "tn": tn.astype(int).tolist(),
        "precision": precision.tolist(),
        "recall_sensitivity": recall.tolist(),
        "specificity": specificity.tolist(),
        "f1": f1.tolist(),
        "balanced_accuracy": balanced_accuracy.tolist(),
        "micro_precision": micro_precision,
        "micro_recall_sensitivity": micro_recall,
        "micro_specificity": safe_div(micro_tn, micro_tn + micro_fp),
        "micro_f1": micro_f1,
        "micro_accuracy": safe_div(micro_tp + micro_tn, micro_tp + micro_fp + micro_fn + micro_tn),
    }


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str],
    thresholds: Sequence[float] | float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    if isinstance(thresholds, (float, int)):
        thresholds_array = np.full(len(classes), float(thresholds), dtype=np.float32)
    else:
        thresholds_array = np.asarray(thresholds, dtype=np.float32)
    counts = confusion_metrics(labels, probs, thresholds_array)

    per_class: dict[str, dict[str, float | int]] = {}
    aurocs: list[float] = []
    aps: list[float] = []
    for index, class_name in enumerate(classes):
        class_labels = labels[:, index].astype(int).tolist()
        class_scores = probs[:, index].astype(float).tolist()
        auroc = roc_auc(class_labels, class_scores)
        ap = average_precision(class_labels, class_scores)
        aurocs.append(auroc)
        aps.append(ap)
        per_class[class_name] = {
            "positive_cases": int(labels[:, index].sum()),
            "negative_cases": int(labels.shape[0] - labels[:, index].sum()),
            "threshold": float(thresholds_array[index]),
            "auroc": auroc,
            "auprc_ap": ap,
            "precision": float(counts["precision"][index]),
            "recall_sensitivity": float(counts["recall_sensitivity"][index]),
            "specificity": float(counts["specificity"][index]),
            "f1": float(counts["f1"][index]),
            "balanced_accuracy": float(counts["balanced_accuracy"][index]),
            "tp": int(counts["tp"][index]),
            "fp": int(counts["fp"][index]),
            "fn": int(counts["fn"][index]),
            "tn": int(counts["tn"][index]),
        }

    macro_f1_values = [float(v) for v in counts["f1"]]
    macro_bacc_values = [float(v) for v in counts["balanced_accuracy"]]
    return {
        "num_cases": int(labels.shape[0]),
        "num_classes": len(classes),
        "thresholds": {class_name: float(thresholds_array[i]) for i, class_name in enumerate(classes)},
        "macro_f1": mean(macro_f1_values) if macro_f1_values else math.nan,
        "macro_balanced_accuracy": mean(macro_bacc_values) if macro_bacc_values else math.nan,
        "macro_auroc": mean(finite(aurocs)) if finite(aurocs) else math.nan,
        "macro_auprc_ap": mean(finite(aps)) if finite(aps) else math.nan,
        "micro_accuracy": counts["micro_accuracy"],
        "micro_f1": counts["micro_f1"],
        "micro_precision": counts["micro_precision"],
        "micro_recall_sensitivity": counts["micro_recall_sensitivity"],
        "micro_specificity": counts["micro_specificity"],
        "per_class": per_class,
    }


def tune_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str],
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    grid = np.linspace(min_threshold, max_threshold, int(steps), dtype=np.float32)
    thresholds: dict[str, float] = {}
    for index, class_name in enumerate(classes):
        class_labels = labels[:, index].astype(bool)
        if class_labels.sum() == 0:
            thresholds[class_name] = 0.5
            continue
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            preds = probs[:, index] >= threshold
            tp = float(np.logical_and(class_labels, preds).sum())
            fp = float(np.logical_and(~class_labels, preds).sum())
            fn = float(np.logical_and(class_labels, ~preds).sum())
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2.0 * precision * recall, precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[class_name] = best_threshold
    return thresholds


def metric_mean_std(values: Sequence[float]) -> tuple[float, float, int]:
    usable = finite(values)
    if not usable:
        return math.nan, math.nan, 0
    return mean(usable), pstdev(usable), len(usable)


def write_prediction_json(
    path: Path,
    *,
    model_name: str,
    checkpoint: str | None,
    csv_path: str,
    volume_ids: Sequence[str],
    labels: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str],
    thresholds: Sequence[float] | dict[str, float] | float = 0.5,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(thresholds, dict):
        threshold_list = [float(thresholds[class_name]) for class_name in classes]
    elif isinstance(thresholds, (float, int)):
        threshold_list = [float(thresholds)] * len(classes)
    else:
        threshold_list = [float(v) for v in thresholds]

    metrics = compute_metrics(labels, probs, classes, threshold_list)
    cases: list[dict[str, Any]] = []
    for row_index, volume_id in enumerate(volume_ids):
        true_values = labels[row_index].astype(int).tolist()
        prob_values = probs[row_index].astype(float).tolist()
        pred_values = [int(prob >= threshold_list[i]) for i, prob in enumerate(prob_values)]
        cases.append(
            {
                "volume_id": volume_id,
                "threshold": dict(zip(classes, threshold_list)),
                "classes": list(classes),
                "true_case_labels": dict(zip(classes, true_values)),
                "case_probabilities": dict(zip(classes, prob_values)),
                "pred_case_labels": dict(zip(classes, pred_values)),
                "probability_max_over_patches": dict(zip(classes, prob_values)),
                "probability_mean_over_patches": dict(zip(classes, prob_values)),
                "probability_top5_patch_mean": dict(zip(classes, prob_values)),
                "positive_patch_count": dict(zip(classes, pred_values)),
            }
        )

    summary: dict[str, Any] = {
        "model": model_name,
        "checkpoint": checkpoint,
        "csv": csv_path,
        "num_cases": len(cases),
        "classes": list(classes),
        "threshold": dict(zip(classes, threshold_list)),
        "metrics": metrics,
        "class_mean_probability": dict(zip(classes, probs.mean(axis=0).astype(float).tolist())),
        "class_true_positive_cases": dict(zip(classes, labels.sum(axis=0).astype(int).tolist())),
        "class_pred_positive_cases": dict(
            zip(classes, (probs >= np.asarray(threshold_list).reshape(1, -1)).sum(axis=0).astype(int).tolist())
        ),
    }
    if extra_summary:
        summary.update(extra_summary)

    payload = {"summary": summary, "cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload

