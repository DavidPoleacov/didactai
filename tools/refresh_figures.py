#!/usr/bin/env python3
"""Regenerate current practice-bank plots from CSV data and the saved report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data/processed/exercises_augmented.csv"
FLAGS_PATH = ROOT / "data/processed/exercise_review_flags.csv"
REPORT_PATH = ROOT / "models/evaluation_report.json"
ASSETS = ROOT / "assets"


def main() -> int:
    from src.exercise_bank import load_practice_bank

    source = pd.read_csv(DATA_PATH)
    practice = load_practice_bank(source, FLAGS_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    practice["Domeniu"].value_counts().sort_values().plot.barh(ax=ax, color="#4F46E5")
    ax.set(title=f"Available practice exercises by domain (n={len(practice):,})", xlabel="Exercises", ylabel="Domain")
    fig.tight_layout()
    fig.savefig(ASSETS / "domain_distribution.png", dpi=160)
    plt.close(fig)

    difficulty_order = ["1 - bază", "2 - mediu", "3 - consolidare", "4 - avansat"]
    counts = practice["Dificultate_group"].value_counts().reindex(difficulty_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts.plot.bar(ax=ax, color="#0F766E")
    ax.set(title="Available practice exercises by labeled difficulty", xlabel="Dataset difficulty label", ylabel="Exercises")
    ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    fig.savefig(ASSETS / "difficulty_distribution.png", dpi=160)
    plt.close(fig)

    metrics = {
        "Structured difficulty": report["structured_model"]["model"]["macro_f1"],
        "Text domain": report["unstructured_model"]["model"]["macro_f1"],
    }
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(metrics.keys(), metrics.values(), color=["#4F46E5", "#0F766E"])
    ax.set_ylim(0, 1)
    ax.set(title="Saved holdout macro-F1 (classification only)", ylabel="Macro-F1")
    ax.bar_label(bars, fmt="%.3f")
    fig.tight_layout()
    fig.savefig(ASSETS / "macro_f1_comparison.png", dpi=160)
    plt.close(fig)
    print(
        f"Regenerated active figures from {len(practice):,} available rows; "
        f"macro-F1 values={metrics}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
