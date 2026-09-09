# TRACE for PatchChestCT

Official release preparation for **TRACE**, a deterministic V-JEPA 2.1-B framework for joint case-level abnormality classification and patch-level localization on PatchChestCT.

This repository contains code only. Patient data, PatchChestCT annotations, pretrained weights, checkpoints, predictions, and experiment logs are intentionally excluded.

## Method at a glance

TRACE starts from the official V-JEPA 2.1-B pretrained encoder and adds four closely connected components:

1. **PASE** uses anatomically contiguous depth grouping and SmoothOR (LogMeanExp) pooling. Strong evidence from a small lesion is retained instead of being diluted by many negative tokens.
2. **Fine supervision** applies BCE + Dice directly to a `24 x 12 x 12` annotation-aligned prediction grid.
3. **GAC** aggregates fine logits to the common `6 x 12 x 12` grid and aligns them with the coarse prediction through a fine-to-coarse consistency loss.
4. **CSEA** replaces hard maximum case readout at inference with fixed-temperature (`tau=0.5`) raw-logit LogMeanExp. CSEA changes case classification only; localization uses the fine-to-coarse map.

The training objective is

```text
L = L_coarse + lambda_fine * L_fine + lambda_gac * L_gac
```

where `L_coarse` and `L_fine` are BCE + Dice losses, `lambda_fine=0.25`, `lambda_gac=0.05`, and both auxiliary weights are linearly warmed up for five epochs.

## Main result

Five-fold results on the locked test set are shown below. Values are percentages and reported as mean ± population standard deviation across folds.

| Method | AUROC | AUPRC | Macro-F1 | BACC | Patch-AUPRC | Patch-DSC |
|---|---:|---:|---:|---:|---:|---:|
| TRACE | 84.12 ± 1.03 | 57.70 ± 1.85 | 53.66 ± 2.48 | 72.87 ± 1.90 | 48.69 ± 0.35 | 50.33 ± 0.38 |

`Patch-DSC` above follows the historical PatchChestCT test-oracle threshold protocol for direct comparability. The CSEA evaluator additionally exports leakage-free `Patch-DSC@ValThr`, where each class threshold is selected on validation data and then frozen on test data.

## Repository layout

```text
TRACE-PatchChestCT/
├── classification_code/
│   ├── patchchestct_grid.py          # legacy and anatomical grid protocols
│   ├── patchchestct_pooling.py       # mean, SmoothOR, fixed resampling
│   ├── train_patchchestct_official_patch_fold0.py
│   ├── evaluate_patchchestct_csea_raw_logits.py
│   ├── patchchestct_vjepa2_1/        # V-JEPA dense prediction model
│   ├── patchchestct_* /              # comparison-backbone interfaces
│   └── tests/
├── scripts/
│   ├── run_cv.py                     # portable deterministic CV launcher
│   └── summarize_cv.py               # five-fold table aggregation
├── docs/
│   ├── DATA.md
│   └── METHOD.md
└── examples/manifests/
```

## Installation

Python 3.10 and a CUDA-capable PyTorch installation are recommended. The experiments reported in the paper used PyTorch 2.5.1 and torchvision 0.20.1.

```bash
conda create -n trace python=3.10 -y
conda activate trace
pip install -r requirements.txt
```

The first V-JEPA run uses `torch.hub` to obtain the official [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) implementation and V-JEPA 2.1-B checkpoint. These external files are not redistributed here.

## Data preparation

Obtain CT-RATE/PatchChestCT through their official access procedure and prepare the official preprocessed CT volumes as `.npz` files. Each CT file must contain `arr_0`; patch annotations are per-disease `.npz` files with shape `24 x 12 x 12`.

Create five-fold manifests in the following layout:

```text
splits/
├── test.csv
├── fold_0/train.csv
├── fold_0/val.csv
├── ...
└── fold_4/val.csv
```

Required columns and an artificial row are provided in [manifest.example.csv](examples/manifests/manifest.example.csv). Paths may be absolute or relative to the manifest file. No patient record is included in the example. See [DATA.md](docs/DATA.md) for the annotation convention.

## Reproduce TRACE

Run all folds sequentially on one GPU:

```bash
python scripts/run_cv.py \
  --variant trace \
  --splits-dir /path/to/splits \
  --output-root /path/to/outputs \
  --folds 0 1 2 3 4 \
  --gpu 0
```

The launcher locks the reported protocol: seed 2026, deterministic PyTorch algorithms, V-JEPA 2.1-B official initialization, 30 epochs, batch size 2, gradient accumulation 4, AdamW at `1e-5`, anatomical `6 x 12 x 12` grid, PASE temperature 1, fine weight 0.25, GAC weight 0.05, and five-epoch warm-up. It then runs CSEA (`tau=0.5`) from the best checkpoint.

To aggregate completed CSEA folds:

```bash
python scripts/summarize_cv.py /path/to/outputs/trace_seed2026_e30_csea_tau0p5
```

## Ablations

The same launcher exposes controlled variants while leaving all other settings unchanged:

```bash
python scripts/run_cv.py --variant baseline  --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
python scripts/run_cv.py --variant pase      --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
python scripts/run_cv.py --variant pase-fine --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
```

Add `--csea` to evaluate any completed variant with the fixed case-level CSEA readout.

## Tests

The unit tests verify anatomical grouping, non-overlapping physical-center partitions, and numerical equivalence of the optimized SmoothOR implementation:

```bash
python -m unittest discover -s classification_code/tests -v
```

## Reproducibility notes

- Every fold starts independently from the same official V-JEPA 2.1-B pretrained weights; no task-finetuned checkpoint is used for initialization.
- Train, validation, and test manifests must remain disjoint. Validation selects the best epoch and per-class case thresholds; test data are used only for final reporting.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabling, seeded workers, and deterministic PyTorch algorithms are enabled by the launcher.
- Output directories are never overwritten.

## Citation

The manuscript is under preparation. Please replace this placeholder with the final bibliographic entry before public release:

```bibtex
@article{trace_patchchestct,
  title   = {TRACE: Joint Case Classification and Patch Localization in Chest CT},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```

## License and data terms

No license is asserted by this preparation folder. Add the license approved by the authors and institution before making the repository public. CT-RATE, PatchChestCT, V-JEPA 2, and comparison backbones remain subject to their respective licenses and data-use agreements.
