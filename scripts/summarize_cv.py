#!/usr/bin/env python3
"""Aggregate fold-level macro metrics exported by the trainer/evaluator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev


METRICS = ("AUROC", "AUPRC", "Macro-F1", "BACC", "Patch-AUPRC")


def parse_value(row: dict[str, str], metric: str) -> float:
    numeric = f"{metric} Mean (%)"
    if numeric in row:
        return float(row[numeric])
    value = row[f"{metric} (%)"].split("±", 1)[0].strip()
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing fold_0 ... fold_4")
    args = parser.parse_args()
    values = {metric: [] for metric in (*METRICS, "Patch-DSC")}
    for fold in range(5):
        path = args.run_dir / f"fold_{fold}" / "summary_metrics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        for metric in METRICS:
            values[metric].append(parse_value(row, metric))
        dsc_name = "Patch-DSC Oracle" if "Patch-DSC Oracle Mean (%)" in row else "Patch-DSC"
        values["Patch-DSC"].append(parse_value(row, dsc_name))

    print("Method\tAUROC\tAUPRC\tMacro-F1\tBACC\tPatch-AUPRC\tPatch-DSC")
    formatted = [f"{mean(values[m]):.2f} ± {pstdev(values[m]):.2f}" for m in values]
    print(f"{args.run_dir.name}\t" + "\t".join(formatted))


if __name__ == "__main__":
    main()
