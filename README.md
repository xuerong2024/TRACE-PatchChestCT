# TRACE for PatchChestCT

Official release preparation for **TRACE**, a deterministic V-JEPA 2.1-B framework for joint case-level abnormality classification and patch-level localization on PatchChestCT.

This repository contains code only. Patient data, PatchChestCT annotations, pretrained weights, checkpoints, predictions, and experiment logs are intentionally excluded.

## Method at a glance

TRACE starts from the official V-JEPA 2.1-B pretrained encoder and adds four closely connected components:

1. **TASE** uses anatomically contiguous depth grouping and SmoothOR (LogMeanExp) pooling. Strong evidence from a small lesion is retained instead of being diluted by many negative tokens.
2. **Fine supervision** applies BCE + Dice directly to a `24 x 12 x 12` annotation-aligned prediction grid.
3. **GAC** aggregates fine logits to the common `6 x 12 x 12` grid and aligns them with the coarse prediction through a fine-to-coarse consistency loss.
4. **CSEA** replaces hard maximum case readout at inference with fixed-temperature (`tau=0.5`) raw-logit LogMeanExp. CSEA changes case classification only; localization uses the fine-to-coarse map.

The training objective is

```text
L = L_coarse + lambda_fine * L_fine + lambda_gac * L_gac
```

where `L_coarse` and `L_fine` are BCE + Dice losses, `lambda_fine=0.25`, `lambda_gac=0.05`, and both auxiliary weights are linearly warmed up for five epochs.

## Evaluation protocol

All reported localization DSC values use **Patch-DSC@ValThr**: one threshold is selected per disease on the validation split and then frozen on the test split. Test-oracle DSC is not used by the trainer, evaluator, or CV summarizer.

The V-JEPA encoder and token-wise classifier are optimized jointly end to end. The paper launchers likewise optimize every comparison encoder end to end; optional freezing switches retained in generic utility modules are not used by any paper configuration. Numerical results are intentionally not bundled with this code-only release; use the locked paper configurations below to reproduce them.

## Repository layout

```text
TRACE-PatchChestCT/
├── classification_code/
│   ├── patchchestct_grid.py          # legacy and anatomical grid protocols
│   ├── patchchestct_pooling.py       # mean, SmoothOR, fixed resampling
│   ├── train_patchchestct_official_patch_fold0.py
│   ├── train_patchchestct_official_case_fold0.py
│   ├── evaluate_patchchestct_csea_raw_logits.py
│   ├── patchchestct_vjepa2_1/        # V-JEPA dense prediction model
│   ├── patchchestct_*/               # comparison-backbone interfaces
│   └── tests/
├── scripts/
│   ├── run_cv.py                     # portable deterministic CV launcher
│   ├── run_joint_cv.py               # paper main-table launcher
│   ├── run_encoder_screening_cv.py   # case-level encoder screening
│   └── summarize_cv.py               # five-fold table aggregation
├── configs/paper/                    # versioned paper protocols
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

Obtain the source CT volumes separately from the official [CT-RATE dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) under its Data Usage Agreement, and prepare the official preprocessed volumes as `.npz` files. Each CT file must contain `arr_0`.

The patch annotations used by this project are **not created or redistributed by this repository**. Download the official dataset [PatchChestCT: A patch-level spatial annotation dataset for nine abnormalities in chest CT](https://zenodo.org/records/19707049) from Zenodo (version DOI: [`10.5281/zenodo.19707049`](https://doi.org/10.5281/zenodo.19707049), CC BY 4.0). The deposit provides `annotations-train.zip` and `annotations-valid.zip`; it contains patch annotations and CT-RATE volume identifiers, but no CT images. The companion official implementation is available at [SadVoxel/PatchChestCT](https://github.com/SadVoxel/PatchChestCT).

After extraction, each manifest `annotation_dir` must point to an existing per-volume annotation directory. Individual disease files are binary `.npz` arrays with shape `24 x 12 x 12`. A missing disease file inside a valid volume directory means that disease is negative, but a missing `annotation_dir` indicates an invalid dataset path and causes the loader to stop with an error rather than silently treating all nine diseases as negative.

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

The launcher locks the reported protocol: seed 2026, deterministic PyTorch algorithms, end-to-end fine-tuning of the official V-JEPA 2.1-B encoder and token classifier, 30 epochs, batch size 2, gradient accumulation 4, AdamW at `1e-5` for both encoder and classifier, anatomical `6 x 12 x 12` grid, TASE temperature 1, fine weight 0.25, GAC weight 0.05, and five-epoch warm-up. Case labels are the spatial OR of the cropped fine-grid annotation. It then runs CSEA (`tau=0.5`) from the best checkpoint.

To aggregate completed CSEA folds:

```bash
python scripts/summarize_cv.py /path/to/outputs/trace_seed2026_e30_csea_tau0p5
```

## Ablations

The same launcher exposes controlled variants while leaving all other settings unchanged:

```bash
python scripts/run_cv.py --variant baseline  --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
python scripts/run_cv.py --variant baseline-fine --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
python scripts/run_cv.py --variant tase      --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
python scripts/run_cv.py --variant tase-fine --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
python scripts/run_cv.py --variant legacy-modulo --splits-dir /path/to/splits --output-root /path/to/outputs --folds 1 --gpu 0 --no-csea
```

Add `--csea` to evaluate any completed anatomically aligned variant with the fixed case-level CSEA readout. Variants with fine supervision automatically report localization through the fine-to-coarse route, while their case aggregation remains Max unless `--csea` is supplied. `baseline-fine` provides fine supervision without TASE or GAC. `legacy-modulo` reproduces the interleaved modulo-six depth grouping used as the deliberately unaligned TASE design control.

## Paper main-table experiments

The exact joint classification/localization settings for R3D-18, Swin3D-T, MViT-v2-S, VoCo-10K, the V-JEPA baseline, and TRACE are versioned in [`configs/paper/joint_cv.json`](configs/paper/joint_cv.json). Run any subset sequentially on one GPU:

```bash
python scripts/run_joint_cv.py \
  --methods r3d18 swin3d-t mvit-v2-s vjepa-baseline trace \
  --splits-dir /path/to/splits \
  --output-root /path/to/outputs \
  --folds 0 1 2 3 4 \
  --gpu 0
```

VoCo additionally requires its external checkpoint:

```bash
python scripts/run_joint_cv.py \
  --methods voco10k \
  --voco-pretrained-checkpoint /path/to/VoCo_10k.pt \
  --splits-dir /path/to/splits \
  --output-root /path/to/outputs \
  --gpu 0
```

The component and TASE-design matrix is recorded in [`configs/paper/trace_ablation.json`](configs/paper/trace_ablation.json) and is executable through `scripts/run_cv.py`.

## Case-level encoder screening

The 12 encoders in the paper's screening table use the official case-level NoisyOR protocol. Their five-fold settings are recorded in [`configs/paper/encoder_screening.json`](configs/paper/encoder_screening.json):

```bash
python scripts/run_encoder_screening_cv.py \
  --methods medicalnet-r18 models-genesis r3d18 vjepa2.1-b \
  --splits-dir /path/to/splits \
  --output-root /path/to/case_outputs \
  --folds 0 1 2 3 4 \
  --gpu 0
```

The screening table contains 12 encoder--initialization configurations. MViT-v2-S is trained from random initialization as an architectural control; the remaining entries use the medical or video initialization recorded in the configuration and run metadata. Screening uses manifest-level case labels and the official case-level NoisyOR route, whereas the joint patch-supervised experiments derive crop-consistent case labels from the cropped fine-grid annotation.

Pretrained weights are not redistributed. Before running the corresponding entries, place MedicalNet weights under `nnunet_data/Bronchidata/PatchChestCT/pretrained_medical_models/MedicalNet/`, VoCo at `nnunet_data/Bronchidata/PatchChestCT/pretrained_weights/voco10k/VoCo_10k.pt`, and the I3D/Slow checkpoints in the standard Torch Hub checkpoint cache. VideoMAE and TimeSformer use complete local Hugging Face snapshots; V-JEPA uses the official PyTorch Hub source and checkpoint. Models Genesis follows its public checkpoint URL when it is not already cached.

## Tests

The unit tests verify anatomical grouping, non-overlapping physical-center partitions, and numerical equivalence of the optimized SmoothOR implementation:

```bash
python -m unittest discover -s classification_code/tests -v
```

## Reproducibility notes

- Every fold starts independently from the same official V-JEPA 2.1-B pretrained weights. The encoder and newly initialized token-wise classifier are trained jointly end to end at `1e-5`; no task-finetuned checkpoint is used for initialization.
- In joint patch-supervised experiments, case labels are recomputed after cropping as a spatial OR over each disease's `24 x 12 x 12` fine-grid target. The separate case-level encoder screening uses manifest-level labels.
- Patch-DSC always means Patch-DSC@ValThr. Each disease threshold is selected on validation and applied unchanged to test.
- Patch AUPRC and Patch-DSC use cells from positive-annotation cases for the corresponding disease, matching the reference patch-evaluation protocol; all-negative cases are excluded from that disease's patch metric.
- Five-fold tables use the population standard deviation across runs (`statistics.pstdev`).
- Train, validation, and test manifests must remain disjoint. Validation selects the best epoch and per-class case thresholds; test data are used only for final reporting.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabling, seeded workers, and deterministic PyTorch algorithms are enabled by the launcher.
- Output directories are never overwritten.

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{trace_patchchestct,
  title   = {TRACE: Token-Aligned Regional Aggregation and Granularity-Consistent Evidence Learning for Multi-Abnormality Chest CT Classification and Patch Localization},
  author  = {Wang, Xuerong and Zhang, Tong and Zhao, Yuzhang and Tian, Yao and Zhao, Tao and Yang, Le and Wang, Binglu},
  journal = {Under review},
  year    = {2026}
}
```

## License and data terms

No open-source license is currently granted for this repository; all rights are reserved unless a `LICENSE` file is added. CT-RATE, PatchChestCT, V-JEPA 2, and comparison backbones remain subject to their respective licenses and data-use agreements.
