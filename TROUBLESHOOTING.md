# Troubleshooting

## Missing dependencies

Create a Python 3.12 environment and install `requirements-light.txt`. The tested app does not require TensorFlow. Check resolved versions with:

```bash
python --version
python -m pip show streamlit pandas numpy scikit-learn joblib matplotlib
```

The saved scikit-learn classifiers require `scikit-learn==1.9.0`; avoid loading them under an incompatible version.

## Startup reports missing resources

`app.py` requires the saved structured and text classifiers, the evaluation report, the augmented CSV, and `data/processed/exercise_review_flags.csv`. It reports failures rather than silently training or bypassing the review filter. Restore the missing tracked file or correct its format/required columns. The app does not require `.streamlit/secrets.toml`; the checked-in `.streamlit/secrets.toml.example` is only a placeholder template.

Run the lightweight diagnostics:

```bash
python diagnose.py
```

TensorFlow is explicitly optional and is reported as skipped by default. To request an actual neural inference check, install `requirements-neural.txt` and run `python diagnose.py --neural`; this fails clearly if TensorFlow or the neural model is unavailable.

## Model mismatch or unexpected metrics

Do not retrain during ordinary startup or to conceal a mismatch. Run `python tools/verify_saved_models.py` to compare the saved classifiers against the deterministic held-out rows and report. Run `python tools/check_generated_math.py` for provenance and content-review counts. Neither utility evaluates arbitrary mathematical expressions.

## App behavior

- Home is a simulation; preview answers and hints must not change Tutor counters or progress.
- Tutor progress records only definite correct/incorrect comparisons. `needs_review`, `invalid_key`, and empty responses do not count as attempts.
- Tutor history and approximate mastery exist only for the current Streamlit session. They are cleared by **Resetează progresul Tutorului**.
- Missing recommendations within a selected domain are shown as an empty-state message; the app does not silently recommend from another domain.
- A review date is informational only; there is no saved schedule or reminder delivery.

## Running the app

```bash
python -m streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. If the page appears old after a code update, stop the local server and launch it again; no Cloud deploy is part of the demo instructions.
