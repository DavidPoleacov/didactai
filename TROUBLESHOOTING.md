# Troubleshooting

Start with the actual exception and environment versions. Configuration files alone do not establish cloud RAM usage, installation time or successful deployment.

## Lightweight startup

Create a fresh Python 3.12 environment, install `requirements-light.txt`, then run `python diagnose.py`, `python -m unittest discover -s tests -v` and `streamlit run app.py`. The optional neural startup test is skipped without TensorFlow; the other tests must pass. Neural inference in `diagnose.py` is opt-in with `--neural`.

## Serialized model warnings

Check that scikit-learn is **1.9.0**, as pinned in both requirement files. A different version can emit compatibility warnings or fail to load the artifacts. Reproduce issues in the deployment environment rather than suppressing warnings.

## Cloud chooses the wrong requirements

Community Cloud recognizes `requirements.txt` and other documented filenames. It does not automatically select `requirements-light.txt`. Put the desired lightweight contents in `requirements.txt` in your deployment branch/copy, as described in `DEPLOYMENT.md`, and inspect the next build log.

## Neural prediction

The optional model is an experimental classifier trained on synthetic interaction labels. Test it separately with `python diagnose.py --neural` after installing full requirements. Successful import or a `.keras` file alone does not establish successful inference.

The audit environment could import TensorFlow but aborted on a basic tensor addition because process-filesystem information was unavailable. The audit verifies the lightweight path and independently checks saved neural weights, not TensorFlow inference. This environment-specific failure does not establish that the model fails on ordinary computers.

## Answer needs review

The checker supports normalized exact strings and constrained rational scalars. It cannot prove general symbolic equivalence or convert units. An uncertain result does not change progress. A message requests review; the app does not create a teacher queue or notify a reviewer.

## A problem disappears

The practice bank excludes entries in `data/processed/exercise_review_flags.csv`. Review the original task, diagrams, key and solution before removing a flag. The training CSV remains unchanged: its 1,344-row count differs from the 1,285-row practice count.

## Progress disappears

Progress lives in Streamlit session state, not a persistent student account. A reset/new session loses it. Review dates are suggestions, not scheduled reminders. A database and real-student validation would be separate work.
