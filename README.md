# DidactAI

DidactAI is an open-source Romanian-language mathematics tutoring prototype built with Streamlit. It combines a content-filtered exercise bank for browsing, a checker-compatible subset for automatically scored practice, predefined hints, rule-based feedback, and two saved exercise classifiers. It is intended for experimentation and learning, not as a replacement for a teacher.

## Capabilities

- Explore problems by domain and topic in Home. Home interactions are **simulations** and do not change Tutor progress.
- Practise an automatically gradable exercise in Tutor AI, request staged stored hints, receive conservative answer feedback, and see a same-domain suggestion after a verified correct answer.
- Review verified Tutor attempts, approximate rule-based mastery, and session-only history in Progress.
- Complete a short diagnostic. Only definite correct/incorrect answers contribute to its domain scores; it is separate from tracked Tutor history.
- Classify exercise domain from problem text and exercise difficulty from structured metadata.

Hints and Romanian reflection questions are predefined. Recommendations use simple domain/difficulty rules and seeded sampling; they do not diagnose misconceptions or use a learned ranking model. The app does not generate explanations with an LLM or analyze reflection responses.

## Installation and quick start

Python 3.12 is the tested runtime. The default `requirements.txt` is lightweight and is also the recognized dependency file for Streamlit Community Cloud. TensorFlow is not required.

**Windows PowerShell or Command Prompt** (use the venv Python directly; activation is optional):

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install --upgrade pip
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-light.txt
.\.venv-demo\Scripts\python.exe -m streamlit run app.py
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install --upgrade pip
.venv-demo/bin/python -m pip install -r requirements-light.txt
.venv-demo/bin/python -m streamlit run app.py
```

Open the local URL shown by Streamlit (normally `http://localhost:8501`). Run the checks before launching:
Pip's official upgrade guidance is at [pip.pypa.io/en/stable/installation](https://pip.pypa.io/en/stable/installation/).

```bash
python -m unittest discover -s tests -v
python diagnose.py
```

`requirements-demo.lock` is a platform-specific Linux/CPython 3.12.3 snapshot and pins Streamlit 1.65.0. The fresh range-based environment used for this repair resolved Streamlit 1.66.0; the lock snapshot was not installed or tested here.

For an optional TensorFlow inference installation, use `requirements-neural.txt`. Neural inference runs only when TensorFlow and the saved neural model are both available. Missing neural artifacts are not trained at startup. Rebuild models only through the explicit development command `python src/train_models.py` (classical models and reports) or `python src/neural_student_state_model.py` (neural model).

## How the interface works

- **Home:** browse all content-eligible exercises and view domain/difficulty predictions. Fifteen checker-incompatible answer formats and one separately reviewed open-choice task remain available with a manual-review label. Simulated answers, hints, and recommendations never write to Tutor tracking. Changing the selected preview exercise clears its answer, hints, feedback, estimate, and recommendation; returning starts a blank simulation.
- **Tutor AI:** choose among automatically gradable exercises or ask for an automatic choice. Definite wrong and correct answers count as separate attempts. Unverified comparisons do not update progress. Duplicate completion is blocked within an exercise run; deliberate retakes in a new run remain allowed.
- **Progress:** counts verified attempts and groups them by the exercise's actual domain. Completed distinct exercises are counted separately. The mastery value is a global approximate rule-based indicator, not a calibrated probability. History is held in the current Streamlit session and is not saved to an account. A displayed review date is only a suggestion; it is not scheduled and sends no reminder.

## Architecture and models

- `app.py` contains the Streamlit interface and session-scoped Tutor workflow.
- `src/exercise_bank.py` applies `data/processed/exercise_review_flags.csv` by normalized `problem_hash`, removes empty statements/keys, and preserves source indices for model features and history.
- `src/pedagogical_engine.py` implements conservative scalar answer comparison, separate key-format and semantic eligibility, hash-validated display-only hint corrections, hint staging, diagnostic scoring, and simple recommendations. Diagnostic scoring accepts explicit verified verdicts; legacy prevalidated inputs require a boolean `correct` field with no `verified` or `status` markers.
- A scikit-learn `RandomForestClassifier` predicts difficulty from metadata: source/item metadata, problem/solution/answer lengths, mathematical-symbol indicators, and categorical topic, known domain, and source type.
- A TF-IDF/Complement Naive Bayes pipeline predicts domain from Romanian problem text.
- An optional TensorFlow model classifies interactions in a synthetic learner-state dataset. It is not used in the lightweight installation and its labels are rule-generated, not real-student observations.

## Dataset and provenance

`data/processed/exercises_augmented.csv` contains 1,344 original training rows: 1,212 generated and 132 source-derived. The content-review flags exclude 59 problems by normalized problem text—42 source-derived items and 17 ambiguous generated savings items—leaving 1,285 content-eligible rows for browsing. Comparing each stored key with itself derives 1,270 format-compatible rows. A separately reviewed semantic exclusion removes open-choice exercise #37 from the automatically gradable subset, leaving 1,269 automatically gradable rows and 16 browsing/manual-review rows (15 format-incompatible rows plus #37). Key round trips test format compatibility only; they do not establish mathematical correctness or teacher approval. The original CSV, flags, model features, saved models, and evaluation reports are unchanged. `data/processed/exercises_processed.csv` is an older 489-row dataset retained for provenance, not the active bank. The neural interaction dataset contains 2,200 synthetic rows.

The display-only quadratic corrections are versioned in `data/processed/hint_display_corrections.csv` with stable source indices, problem hashes, original/corrected steps, rationale, and source-step SHA-256. Arithmetic validation covers the reviewed discriminant substitutions, not the whole bank.

The saved classical holdout metrics are:

| Classifier | Holdout rows | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Structured difficulty | 296 | 0.922297 | 0.831212 |
| Text domain | 279 | 0.982079 | 0.974313 |

The saved synthetic neural test split contains 440 rows and reports accuracy 0.922727 / macro-F1 0.914048. These are classification metrics, not learning-outcome measures. The exercise data contains related template families across a random row split; metrics may therefore reflect family overlap. External evaluation on independently collected and teacher-reviewed data is still needed. Audit prompts and stored keys without executing input expressions using `python tools/check_generated_math.py`; independently reproduce the saved classical holdouts with `python tools/verify_saved_models.py`.

The active exercise-population plots under `assets/` show the 1,285 content-eligible browsing rows by domain and labeled difficulty; they do not depict the 1,269 automatically gradable Tutor subset. The saved holdout chart shows model classification metrics, not the current practice-bank population. Regenerate only the affected population charts with `python tools/refresh_figures.py --population-only`; older charts, if present under `assets/legacy/`, are historical.

## Docker

The lightweight image uses Python 3.12 without TensorFlow. From the repository root, run `bash run_docker.sh`; do not rely on executable permissions being preserved when extracting the ZIP.

## Testing and maintenance

```bash
python -m unittest discover -s tests -v
python diagnose.py
python tools/check_generated_math.py
python tools/verify_saved_models.py
```

Training utilities are explicit development operations and can overwrite processed data, model files, and reports. They are not invoked by app startup, diagnostics, tests, or the demo.

## Current limitations

- The checker supports exact normalized text and exact rational scalar equivalence with matching variable labels and units. It does not convert units or establish arbitrary symbolic equivalence, proofs, sets, intervals, or ambiguous prose.
- Some bank content is incomplete, ambiguous, or missing diagrams. The 59 content exclusions, 15 key-format exclusions, and one semantic grading exclusion are separate safeguards, not teacher certification of the remaining bank.
- Mastery/recommendations are simple rules; the optional neural model predicts synthetic interaction labels.
- There are no saved accounts, persistent progress, automated reminders, reflection analysis, or measured real-student outcomes.
- Model holdouts are internal random splits, not independent external evaluations.

## Contributors

- **David Poleacov — backend**
- **Beatrice Batrinu — frontend**

AI-assisted maintenance is recorded as later repository history; it is not represented as an original competition feature.
