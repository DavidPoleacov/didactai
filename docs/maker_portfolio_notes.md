# Historical project notes (archived; not current user documentation)

> This file preserves application-era preparation notes only. It is not a current README, not a source of current feature or metric claims, and is not linked from the public project README. Later maintenance is not represented as original competition work.

## Project Description

DidactAI is a Romanian-language mathematics tutoring prototype for Moldovan middle- and high-school students. It combines a curated exercise bank, stored solution steps, rule-based pedagogical feedback, and two scikit-learn classifiers. The current repository also includes an optional neural classifier trained on synthetic student-interaction examples.

This portfolio describes the repository as it exists now. The conservative answer-checking changes, neural-startup fix, regression tests, and portfolio documentation in this revision were made after the original competition version. Do not imply that these later changes were present during the competition.

## Team Contributions

- David Poleacov worked on the backend.
- Beatrice Batrinu worked on the frontend.

These attributions come from the project authors’ description, not from commit metadata. They do not identify ownership of particular commits or files.

## Technical Decisions Visible In The Repository

- The exercise bank and student-facing Streamlit interface remain in Romanian for the intended Moldovan audience.
- The structured difficulty model is a scikit-learn random forest with numeric imputation/scaling and categorical imputation/one-hot encoding. It predicts one of four difficulty groups from exercise metadata and engineered features.
- The text-domain model is TF-IDF plus `ComplementNB`; it classifies the problem statement into a curriculum domain.
- Hints are selected from stored solution steps and a small rule set. Reflective questions are chosen from a fixed list. Mastery updates, diagnostic summaries, and most exercise recommendations are rule-based.
- The optional dense TensorFlow model predicts learner-state labels from nine interaction features. Its 2,200 training examples and labels are synthetic and rule-generated, not real student records.
- The answer checker now accepts exact normalized text and exact rational equivalence for a constrained scalar format. It sends uncertain symbolic, multi-part, unit, and free-text cases for review rather than guessing. It does not execute student input.
- Progress and interaction records live in Streamlit session state and are not persisted to an account or database.

The checked-in `models/evaluation_report.json` is for the 1,344-row augmented exercise dataset. It reports holdout macro-F1 0.831 (structured difficulty; 296 test rows) and 0.974 (text domain; 279 test rows), compared with baselines 0.163 and 0.078. `models/neural_student_state_report.json` reports macro-F1 0.914 on a 440-row test split from synthetic, rule-labeled data. These are classification metrics, not evidence of improved student learning.

## Limitations And Next Steps

- The exercise bank is modest and class distributions are uneven; some prompts may depend on missing diagrams.
- Automatically normalized topic labels may contain errors. External evaluation and teacher review are still needed.
- There is no real-student interaction dataset or evidence of learning outcomes. Do not treat synthetic neural-model metrics as student evidence.
- The answer checker does not implement general symbolic equivalence, unit conversion, set/interval comparison, multi-answer grading, or proof grading. Uncertain cases require a person to review them.
- Progress is session-only. Persistent student accounts and storage are not implemented.
- The saved scikit-learn artifacts load in the inspected environment but warn that they were serialized with scikit-learn 1.9.0 and are being loaded with 1.9.1. Dependency ranges are not exact pins.
- Possible next work: teacher-reviewed answer formats and labels, consented and anonymized interaction data with a documented collection protocol, evaluation on a held-out source, a reproducible dependency lock, and a clearer missing-diagram workflow.

## Documenting The Development Process

Use the repository’s actual history and materials; do not invent stages, testimonials, student outcomes, awards, or earlier screenshots. The visible history includes June 2026 commits such as `a15a53a` (neural student-state integration), `b7d8ab2` (Python/dependency compatibility changes), and `dd47559` (Docker/startup and model-data updates). Before using any commit in a personal narrative, inspect it with `git show --stat <commit>` and `git show <commit>`, then ask the authors who contributed and whether it predates or postdates the competition build.

Existing materials include the README files, `docs/model_report.md`, `docs/dataset_analysis.md`, `docs/judging_lessons.md`, and `docs/exercises_augmented_integration.ipynb`. Check their dates and claims against the current CSVs, JSON reports, and Git history before presenting them as contemporaneous evidence. Capture any new screenshots or recordings yourself from a clearly identified current build.

## Demonstration Script (About 1:50)

Narrate in English while leaving the learner-facing interface and exercise text in Romanian.

**0:00–0:15** — “DidactAI is a Romanian-language mathematics practice prototype for Moldovan students. The exercise bank and learner-facing prompts are in Romanian; I’ll explain the backend in English.”

**0:15–0:35** — In the home/demo area, choose domain `Funcții` and the bank item `Fie f: R → R, f(x) = 4x + 12. Calculați f(0).` The stored answer is `12`. Point out the live text-domain and structured-difficulty predictions, and explain that these are classifiers, not generated explanations.

**0:35–0:55** — Open the hint control and show one progressive hint and the reflective question. Explain that these come from stored solution steps and predefined prompts, selected by pedagogical rules.

**0:55–1:15** — Enter `12` and submit. Show the confirmed feedback and next-exercise recommendation. Explain that mastery changes only for a verified correct or incorrect scalar result; uncertain or empty responses are not recorded as mistakes.

**1:15–1:35** — “The backend uses a random forest for difficulty and TF-IDF with ComplementNB for domain classification. Rules handle hints, mastery updates, and candidate filtering. Progress stays in this browser session.”

**1:35–1:50** — “The current checker is deliberately limited: it compares exact answers and supported rational scalars, while symbolic equivalence and unit conversion need review. The optional neural model uses synthetic labels, so its score is not evidence about real students or learning gains. These checker and portfolio updates are post-competition improvements.”

Keep the full run under two minutes; skip loading screens and avoid implying the optional model is available unless it is visibly active in the environment being recorded.

## Capture Yourself

- A fresh-start view showing the Romanian interface and exercise bank.
- The selected function exercise with its live domain/difficulty predictions.
- A hint and reflective question visible together.
- A confirmed answer with its next-exercise recommendation.
- A needs-review response showing informational feedback and unchanged progress.
- Optionally, the session progress view and the dependency/model status used for the recording.

Do not present newly captured images as historical competition screenshots. Avoid including names, secrets, or personal student information.

## Author Input Needed

- The recorded contributor roles are David Poleacov — backend and Beatrice Batrinu — frontend.
- Which exact commit, code snapshot, and features correspond to the competition submission, and the competition date.
- Confirmation of the dataset source/permission and any manual curation not documented in the repository.
- Whether any real students ever used the prototype; if so, what evidence and consent/data-handling records exist. No such evidence is established by this repository.
- Any author-verified design decisions or development experiences to include in a first-person portfolio narrative.