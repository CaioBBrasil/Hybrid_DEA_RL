# Dataset preparation

`merge_acai_with_negatives.py` builds a one-class YOLO dataset named `acai` by
combining the positive açaí dataset with exactly 1,000 images from Multiple
Tree Detection treated as negative examples.

```bash
python scripts/merge_acai_with_negatives.py
```

Negative images are assigned proportionally to the positive splits. With the
current source data this yields 705 training, 202 validation, and 93 test
negatives. Selection is reproducible with seed 42.

The default destination is `dataset_merged/`. The script refuses to overwrite
an existing destination unless explicitly requested:

```bash
python scripts/merge_acai_with_negatives.py --overwrite
```

The output contains `data.yaml`, `manifest.csv`, `summary.json`, and
`SOURCES.md`. Source datasets under `parts_dataset/` are never modified.

Treating the second dataset as negative requires a separate semantic audit for
the absence of açaí palms and possible source-domain confounding.
