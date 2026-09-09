# Data and manifest format

## CT volumes

The strict protocol expects the official CT-RATE preprocessed `.npz` volumes. Each file contains a 3-D array named `arr_0`, stored in the intensity convention used by PatchChestCT. The loader multiplies it by 1000 to recover Hounsfield units, clips to `[-1000, 200]`, pads/crops to `120 x 240 x 240`, applies the official orientation transform, and uses a `96 x 192 x 192` crop.

NIfTI/DICOM conversion and patient data are deliberately outside this repository. Do not commit converted volumes, patient identifiers, reports, or private access credentials.

## Patch annotations

For each case, `annotation_dir` contains zero or more disease files:

```text
annotation_dir/
├── arterial_wall_calcification.npz
├── pericardial_effusion.npz
└── ...
```

Each present file contains a binary `arr_0` of shape `24 x 12 x 12`. A missing file is treated as an all-zero patch mask. A positive case-level label must have a corresponding annotation file.

## Manifests

`volume_id`, `split`, `image_path`, `annotation_dir`, and the nine `<disease>_label` columns are required by the trainer. Patch-count and bookkeeping columns are retained in the example because they are useful for auditing but are not required by the loader.

The five CV runs use fold-specific training and validation manifests and one locked test manifest:

```text
splits/fold_0/train.csv
splits/fold_0/val.csv
...
splits/fold_4/train.csv
splits/fold_4/val.csv
splits/test.csv
```

Never construct folds by reading the test labels during training. Case thresholds and the optional deployment Patch-DSC threshold are selected on validation data only.
