#!/usr/bin/env python3
"""Check required DidactAI assets; run the optional neural check only with --neural."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_diagnostics(include_neural: bool = False) -> int:
    failures = []
    try:
        import joblib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
        import streamlit  # noqa: F401
        print("PASS required Python dependencies")
    except ImportError as exc:
        failures.append(f"Required dependency is missing: {exc}")
        print(f"FAIL {failures[-1]}")

    required = [
        ROOT / "models/structured_difficulty_model.joblib",
        ROOT / "models/unstructured_domain_model.joblib",
        ROOT / "models/evaluation_report.json",
        ROOT / "data/processed/exercises_augmented.csv",
        ROOT / "data/processed/exercise_review_flags.csv",
        ROOT / ".streamlit/config.toml",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        failures.extend(f"Required file missing: {name}" for name in missing)
        for failure in failures:
            print(f"FAIL {failure}")
    else:
        print("PASS required dataset, review flags, saved models, report, and configuration files")

    if not failures:
        try:
            import pandas as pd

            from src.exercise_bank import load_practice_bank
            from src.model_utils import load_assets, predict_domain_from_text, predict_structured_difficulty

            structured, domain_model, source, report = load_assets()
            practice = load_practice_bank(
                source, ROOT / "data/processed/exercise_review_flags.csv",
            )
            if len(source) != 1344 or len(practice) != 1285:
                print(f"INFO dataset rows: source={len(source)}, practice={len(practice)} (content differs from audited snapshot)")
            sample = practice.loc[[practice.index[0]]]
            predict_domain_from_text(domain_model, str(sample.iloc[0]["Problema"]))
            feature_cols = [
                "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
                "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
                "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type",
            ]
            predict_structured_difficulty(structured, sample[feature_cols])
            if report.get("dataset", {}).get("rows_total") != len(source):
                print("WARN saved training report row count differs from current source data")
            print(f"PASS loaded saved classifiers and made sample inferences; practice rows={len(practice)}")
        except Exception as exc:
            failures.append(f"Required saved assets could not be used: {type(exc).__name__}: {exc}")
            print(f"FAIL {failures[-1]}")

    if include_neural:
        try:
            from src.neural_student_state_model import tf, predict_neural_student_state

            if tf is None:
                raise RuntimeError("TensorFlow is not installed.")
            neural_paths = [
                ROOT / f"models/{name}" for name in (
                    "neural_student_state_model.keras",
                    "neural_student_state_scaler.joblib",
                    "neural_student_state_label_encoder.joblib",
                )
            ]
            missing_neural = [str(path.relative_to(ROOT)) for path in neural_paths if not path.is_file()]
            if missing_neural:
                raise FileNotFoundError(f"Optional neural artifacts are missing: {', '.join(missing_neural)}")
            result = predict_neural_student_state({
                "time_spent_seconds": 45.0, "hint_count": 1.0, "attempt_count": 2.0,
                "is_correct": 0.0, "mistake_count": 1.0, "exercise_difficulty_encoded": 2.0,
                "previous_mastery": 0.55, "consecutive_errors": 1.0, "help_level_requested": 1.0,
            })
            print(f"PASS requested TensorFlow inference: {result['predicted_state']}")
        except Exception as exc:
            failures.append(f"Requested optional neural inference failed: {type(exc).__name__}: {exc}")
            print(f"FAIL {failures[-1]}")
    else:
        print("SKIP optional TensorFlow inference (rerun with --neural to request it)")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true", help="Require and run the optional TensorFlow inference check.")
    arguments = parser.parse_args()
    return run_diagnostics(include_neural=arguments.neural)


if __name__ == "__main__":
    raise SystemExit(main())
