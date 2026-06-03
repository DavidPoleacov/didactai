"""Train the two real ML services required by the ONIA rubric.

Service 1 (structured data): predicts exercise difficulty from tabular metadata and
engineered numeric/categorical features.
Service 2 (unstructured data): predicts curriculum domain from raw problem text.

Both services include baselines, train/test evaluation, grid search, CV, confusion
matrices, and saved artifacts. Metrics are computed, never hardcoded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .data_prep import AUGMENTED_RAW_PATH, build_augmented_dataset, build_processed_dataset
except ImportError:  # allows running as: python src/train_models.py
    from data_prep import AUGMENTED_RAW_PATH, build_augmented_dataset, build_processed_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
ASSETS_DIR = ROOT / "assets"

STRUCTURED_FEATURES_NUM = [
    "Itemul",
    "Sursa_year",
    "problem_chars",
    "problem_words",
    "steps_chars",
    "answer_chars",
    "n_digits",
    "n_math_symbols",
    "has_percent",
    "has_geometry_word",
    "has_equation_word",
    "has_radical",
    "has_function_word",
    "has_real_life_context",
]
STRUCTURED_FEATURES_CAT = ["Tema_norm", "Domeniu", "Sursa_type"]


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _serializable_report(y_true, y_pred) -> Dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # Convert NumPy scalars to plain Python.
    return json.loads(json.dumps(report))


def _confusion_payload(y_true, y_pred) -> Dict:
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "matrix": cm.tolist()}


def _sample_errors(frame: pd.DataFrame, y_true, y_pred, text_col: str = "Problema", n: int = 8):
    errors = []
    for idx, true, pred in zip(frame.index, y_true, y_pred):
        if true != pred:
            row = frame.loc[idx]
            errors.append(
                {
                    "problem": str(row.get(text_col, ""))[:240],
                    "true": str(true),
                    "predicted": str(pred),
                    "tema": str(row.get("Tema_norm", "")),
                    "domeniu": str(row.get("Domeniu", "")),
                    "difficulty": str(row.get("Dificultate_group", "")),
                }
            )
        if len(errors) >= n:
            break
    return errors


def train_structured(data: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    # Strictly structured/metadata features: no raw problem text. Drop unknown targets.
    df = data[data["Dificultate_group"] != "Necunoscut"].copy()
    df = df.drop_duplicates(subset=["problem_hash", "Dificultate_group"])

    y = df["Dificultate_group"]
    X = df[STRUCTURED_FEATURES_NUM + STRUCTURED_FEATURES_CAT]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                STRUCTURED_FEATURES_NUM,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                STRUCTURED_FEATURES_CAT,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=1)),
        ]
    )

    param_grid = {
        "clf__n_estimators": [120],
        "clf__max_depth": [8, None],
        "clf__min_samples_leaf": [1, 3],
        "clf__class_weight": ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)

    report = {
        "task": "Structured difficulty classification",
        "target": "Dificultate_group",
        "inputs": {"numeric": STRUCTURED_FEATURES_NUM, "categorical": STRUCTURED_FEATURES_CAT},
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": y.value_counts().to_dict(),
        "baseline": _metrics(y_test, baseline_pred),
        "model": _metrics(y_test, pred),
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "classification_report": _serializable_report(y_test, pred),
        "confusion_matrix": _confusion_payload(y_test, pred),
        "sample_errors": _sample_errors(df.loc[X_test.index], y_test, pred),
    }
    return best, report


def train_unstructured(data: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    # Unstructured service: raw text -> curriculum domain. Drop exact duplicate text to reduce leakage.
    df = data[data["Domeniu"].notna() & (data["Domeniu"] != "Altele")].copy()
    df = df.dropna(subset=["Problema"])
    df = df.drop_duplicates(subset=["problem_hash", "Domeniu"])
    # Keep classes with at least 8 examples for robust stratification.
    counts = df["Domeniu"].value_counts()
    keep = counts[counts >= 8].index
    df = df[df["Domeniu"].isin(keep)].copy()

    X = df["Problema"].astype(str)
    y = df["Domeniu"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )
    test_frame = df.loc[X_test.index]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b[\w\-]+\b|[√≤≥<>+=/*^%-]",
                ),
            ),
            (
                "clf",
                ComplementNB(),
            ),
        ]
    )
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [3000, 6000],
        "tfidf__min_df": [1, 2],
        "clf__alpha": [0.2, 0.5, 1.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)

    report = {
        "task": "Unstructured text domain classification",
        "target": "Domeniu",
        "input": "raw problem text",
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": y.value_counts().to_dict(),
        "baseline": _metrics(y_test, baseline_pred),
        "model": _metrics(y_test, pred),
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "classification_report": _serializable_report(y_test, pred),
        "confusion_matrix": _confusion_payload(y_test, pred),
        "sample_errors": _sample_errors(test_frame, y_test, pred),
    }
    return best, report


def summarize_dataset(data: pd.DataFrame) -> Dict:
    labeled = data[data["Dificultate_group"] != "Necunoscut"].copy()
    return {
        "rows_total": int(len(data)),
        "rows_with_difficulty_and_topic": int((data["Dificultate"].notna() & data["Tema"].notna()).sum()),
        "exact_duplicate_problem_rows": int(data.duplicated("problem_hash").sum()),
        "missing_by_column": {k: int(v) for k, v in data.isna().sum().to_dict().items()},
        "difficulty_distribution": labeled["Dificultate_group"].value_counts().to_dict(),
        "domain_distribution": data["Domeniu"].value_counts().to_dict(),
        "topic_top_20": data["Tema_norm"].value_counts().head(20).to_dict(),
        "problem_length_words": {
            "min": float(data["problem_words"].min()),
            "median": float(data["problem_words"].median()),
            "mean": float(data["problem_words"].mean()),
            "max": float(data["problem_words"].max()),
        },
        "dataset_decision": "Usable for a strong competition MVP because it contains Romanian math problem text, solutions, topic labels, and difficulty labels. Not production-grade: it is small, topic labels were noisy before normalization, and there are exact duplicates/missing labels; the code explicitly cleans these and reports the limitations.",
    }


def train_all() -> Dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Always refresh the legacy CSV and the augmented workbook output so both paths are available.
    build_processed_dataset(save=True)
    augmented_data = build_augmented_dataset(save=True)
    data = augmented_data if not augmented_data.empty else build_processed_dataset(save=True)
    structured_model, structured_report = train_structured(data)
    unstructured_model, unstructured_report = train_unstructured(data)

    joblib.dump(structured_model, MODELS_DIR / "structured_difficulty_model.joblib")
    joblib.dump(unstructured_model, MODELS_DIR / "unstructured_domain_model.joblib")

    report = {
        "dataset": summarize_dataset(data),
        "structured_model": structured_report,
        "unstructured_model": unstructured_report,
    }
    with open(MODELS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    schema = {
        "structured_numeric_features": STRUCTURED_FEATURES_NUM,
        "structured_categorical_features": STRUCTURED_FEATURES_CAT,
        "structured_target": "Dificultate_group",
        "unstructured_input": "Problema",
        "unstructured_target": "Domeniu",
    }
    with open(MODELS_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    r = train_all()
    print(json.dumps({
        "structured_model": r["structured_model"]["model"],
        "structured_baseline": r["structured_model"]["baseline"],
        "unstructured_model": r["unstructured_model"]["model"],
        "unstructured_baseline": r["unstructured_model"]["baseline"],
    }, indent=2, ensure_ascii=False))
