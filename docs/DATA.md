# Data and manifest format

## CT volumes

The strict protocol expects the official CT-RATE preprocessed `.npz` volumes at `1.5 x 1.5 x 3.0` mm in physical `x-y-z` order, corresponding to `3.0 x 1.5 x 1.5` mm in array `D-H-W` order. Each file contains a 3-D array named `arr_0`, stored in the intensity convention used by PatchChestCT. The loader multiplies it by 1000 to recover Hounsfield units, clips to `[-1000, 200]`, pads/crops to `120 x 240 x 240` in `D-H-W` order, applies the official orientation transform, and uses a `96 x 192 x 192` crop.

NIfTI/DICOM conversion and patient data are deliberately outside this repository. Do not commit converted volumes, patient identifiers, reports, or private access credentials.

## Patch annotations

For each case, `annotation_dir` contains zero or more disease files:

```text
annotation_dir/
├── arterial_wall_calcification.npz
├── pericardial_effusion.npz
└── ...
```

Each present file contains a binary `arr_0` of shape `24 x 12 x 12`. A missing disease file is treated as an all-zero patch mask and therefore produces a negative crop-level case target.

## Manifests

For joint patch-supervised training and evaluation, only `volume_id`, `split`,
`image_path`, and `annotation_dir` are required. Patch-count columns and other
bookkeeping columns are retained in the example for auditing but are not used
as learning targets.

In joint patch-supervised experiments, the `<disease>_label` values are metadata
only. The effective case target is computed after cropping by taking the spatial
OR of the corresponding `24 x 12 x 12` fine-grid annotation. The separate
case-level encoder-screening trainer does use the nine `<disease>_label` columns
as its case targets and therefore requires them in its manifests.

The five CV runs use fold-specific training and validation manifests and one locked test manifest:

```text
splits/fold_0/train.csv
splits/fold_0/val.csv
...
splits/fold_4/train.csv
splits/fold_4/val.csv
splits/test.csv
```

Never construct folds by reading test annotations during training. Case thresholds and Patch-DSC thresholds are selected on validation data only and then frozen on test.
