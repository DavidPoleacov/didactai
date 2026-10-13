# Dataset Notes

This file distinguishes the two processed exercise datasets. Counts were checked against the CSV files in this repository.

| File | Rows | Rows with both topic and difficulty | Repeated normalized problem text |
| --- | ---: | ---: | ---: |
| `data/processed/exercises_processed.csv` (legacy) | 489 | 477 | 51 |
| `data/processed/exercises_augmented.csv` (augmented) | 1,344 | 1,344 | 0 |

The saved exercise classifiers and `models/evaluation_report.json` use the augmented dataset, not the legacy one. The structured classifier uses 1,344 examples. The text classifier filters out the `Altele` domain and duplicate text/domain pairs, leaving 1,265 examples before its train/test split. See `docs/model_report.md` for metric provenance and limitations.
