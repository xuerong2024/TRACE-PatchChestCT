# Method details

## 1. Shared dense classifier

For a `96 x 192 x 192` CT crop, V-JEPA 2.1-B receives a resized `64 x 384 x 384` volume and produces a `32 x 24 x 24` grid of dense tokens. One shared linear classifier maps every token to 18 PatchChestCT logits. Nine annotated diseases are selected by indices `1, 3, 4, 5, 6, 8, 10, 15, 16`.

There are not separate classifiers for the coarse and fine branches. Both routes use the same native-token logits, so improvements cannot come from adding an independent high-capacity head.

## 2. PASE

The annotation grid has 24 ordered depth planes. The legacy official reshape places planes `0,6,12,18` into one coarse depth cell, which mixes distant anatomy. The anatomical protocol instead groups consecutive planes:

```text
[0..3], [4..7], [8..11], [12..15], [16..19], [20..23].
```

Native V-JEPA depth tokens are partitioned into six contiguous physical-center bins:

```text
[0..4], [5..10], [11..15], [16..20], [21..26], [27..31].
```

Within each 3-D bin, PASE uses stable LogMeanExp:

```text
m + tau * log(mean(exp((x - m) / tau))),  m = max(x), tau = 1.
```

It approaches mean pooling at high temperature and max pooling at low temperature while remaining smooth and differentiable.

## 3. Fine supervision

The same native logits are sampled at fixed physical cell centers to obtain a `24 x 12 x 12` grid. Fine supervision is

```text
L_fine = BCEWithLogits(fine_logits, fine_targets)
       + Dice(sigmoid(fine_logits), fine_targets).
```

The resampling matrix is deterministic and follows the `align_corners=False` cell-center rule.

## 4. GAC

Fine logits are aggregated with the same SmoothOR operator to `6 x 12 x 12`. GAC compares the resulting probabilities with a stop-gradient coarse teacher:

```text
L_gac = MSE(sigmoid(SmoothOR(fine_logits)), stopgrad(sigmoid(coarse_logits))).
```

This regularizes the fine route without forcing the coarse branch to chase unstable fine predictions.

## 5. Objective and inference

The reported model uses

```text
L = L_coarse + 0.25 * L_fine + 0.05 * L_gac.
```

The two auxiliary weights ramp linearly from zero over the first five epochs. `L_coarse` alone is used as the validation checkpoint-selection loss, which preserves comparability with the baseline.

At inference:

- Case probability: `sigmoid(LogMeanExp(coarse raw logits, tau=0.5))` (CSEA).
- Localization probability: `sigmoid(SmoothOR(fine raw logits -> 6x12x12, tau=1))`.

CSEA therefore does not alter localization metrics.
