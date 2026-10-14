#!/usr/bin/env python3
"""Recompute saved holdout metrics from the checked-in rows without retraining."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data/processed/exercises_augmented.csv"
STRUCTURED_FEATURES = [
    "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
    "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
    "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type",
]


def scores(expected, predicted) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
    }


def main() -> int:
    dataset = pd.read_csv(DATA_PATH)
    report = json.loads((MODEL_DIR / "evaluation_report.json").read_text(encoding="utf-8"))
    structured = joblib.load(MODEL_DIR / "structured_difficulty_model.joblib")
    domain = joblib.load(MODEL_DIR / "unstructured_domain_model.joblib")
    results = {}

    structured_rows = dataset[dataset["Dificultate_group"] != "Necunoscut"].copy()
    X_train, X_test, _, y_test = train_test_split(
        structured_rows[STRUCTURED_FEATURES],
        structured_rows["Dificultate_group"],
        test_size=0.22,
        random_state=42,
        stratify=structured_rows["Dificultate_group"],
    )
    structured_metrics = scores(y_test, structured.predict(X_test))
    results["structured_model"] = {
        "rows": len(X_test),
        "saved_test_rows": report["structured_model"]["test_rows"],
        **structured_metrics,
        "saved": report["structured_model"]["model"],
    }

    text_rows = dataset[dataset["Domeniu"].notna() & (dataset["Domeniu"] != "Altele")].copy()
    text_rows = text_rows.dropna(subset=["Problema"]).drop_duplicates(subset=["problem_hash", "Domeniu"])
    try:
        _, X_test, _, y_test = train_test_split(
            text_rows["Problema"].astype(str),
            text_rows["Domeniu"].astype(str),
            test_size=0.22,
            random_state=42,
            stratify=text_rows["Domeniu"].astype(str),
        )
    except ValueError:
        _, X_test, _, y_test = train_test_split(
            text_rows["Problema"].astype(str),
            text_rows["Domeniu"].astype(str),
            test_size=0.22,
            random_state=42,
            stratify=None,
        )
    text_metrics = scores(y_test, domain.predict(X_test))
    results["unstructured_model"] = {
        "rows": len(X_test),
        "saved_test_rows": report["unstructured_model"]["test_rows"],
        **text_metrics,
        "saved": report["unstructured_model"]["model"],
    }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    errors = []
    for name, output in results.items():
        saved = output["saved"]
        if output["rows"] != output["saved_test_rows"]:
            errors.append(f"{name}: test row count mismatch")
        for metric in ("accuracy", "macro_f1"):
            if abs(output[metric] - saved[metric]) > 1e-10:
                errors.append(f"{name}: recomputed {metric} differs from report")
    if errors:
        print("VERIFICATION FAILED: " + "; ".join(errors))
        return 1
    print("PASS holdout metrics independently reproduced from saved classifiers and deterministic test splits.")
    print("Note: this reproduces split metrics; it does not establish external generalization or learning outcomes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
