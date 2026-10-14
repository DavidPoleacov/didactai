# Dataset provenance and practice exclusions

## Current exercise bank

`data/processed/exercises_augmented.csv` contains 1,344 labeled rows. The current provenance labels identify 1,212 generated exercises and 132 source-derived exercises. The raw augmented CSV snapshot supplied for this repair has SHA-256 `5e4d71d7cf583d7dffe58fd2828355af417ef19adbd004a0317a5d5e01b12f8d`.

`data/processed/exercise_review_flags.csv` has 59 content-review exclusions, keyed by the normalized `problem_hash`, not row number:

- 42 source-derived problems with missing context, malformed/incorrect keys, incomplete tasks, or proof/diagram limitations.
- 17 ambiguous generated savings questions.

If the CSV remains unchanged, 1,344 − 59 = **1,285 content-eligible browsing rows**. `src/exercise_bank.py` applies the flags while preserving original DataFrame indices for classifier feature lookup and tracked Tutor history. The `row_id` values in the flags record audit provenance only.

Tutor practice, diagnostic questions, and recommendations apply a separate runtime eligibility test: the existing conservative checker must return `correct` when a stored key is compared with itself. On the unchanged dataset, that derives **1,270 automatically gradable rows** and **15 browsing/manual-review rows** at stable source indices 24, 89, 90, 93, 98, 100, 101, 108, 111, 361, 382, 392, 406, 412, and 427. These are checker-format limitations (including radical, multipart, unfamiliar-unit, and explanatory-parenthesis formats), not findings that the underlying mathematics is incorrect. No keys or source rows are rewritten, and the indices are not hardcoded as the runtime filter.

The exercise-population plots in `assets/` continue to describe the 1,285 content-eligible browsing rows; they do not describe the 1,270 automatically gradable subset. This repair does not regenerate those figures because their plotted population is unchanged.

These flags capture identified review limitations. They are not a claim that remaining items were comprehensively verified by teachers. No source answer key is invented or repaired by the loader. The raw workbooks, processed training CSVs, saved models, and original evaluation reports remain training/evaluation artifacts and are not rewritten by normal app use.

Audit the current files using:

```bash
python tools/check_generated_math.py
```

The utility reports generated/source counts and categorized flagged limitations without evaluating any expression. It deliberately does not claim that every unflagged exercise was mathematically checked. Matching a stored answer back to itself verifies format compatibility only, not mathematical correctness.

## Legacy data

`data/processed/exercises_processed.csv` is the retained older 489-row processed set. It is not the active practice bank and its counts/metrics should be treated as legacy. The saved classical models and evaluation report use the 1,344-row augmented dataset.

## Neural interaction data

`data/processed/student_interactions.csv` contains 2,200 synthetic interaction rows used by the optional neural learner-state classifier. These are not real learner records; their labels are rule-generated and do not support claims about actual student outcomes.

## Evaluation boundaries

The classical classifier holdouts use random rows, not groups held out by source or template family. Similar generated templates may therefore overlap between training and test data. The saved classification metrics are not external evaluation and are not learning-outcome evidence. Independent, teacher-reviewed data is needed for those claims. See [model_report.md](./model_report.md) for the saved metrics and split details.
