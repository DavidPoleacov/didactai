"""Load the safe-to-practice subset of the exercise bank."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class ExerciseBankError(ValueError):
    """Raised when the exercise bank or its review flags are unusable."""


def _has_content(values: pd.Series) -> pd.Series:
    cleaned = values.fillna("").astype(str).str.strip()
    return cleaned.ne("") & ~cleaned.str.casefold().isin({"nan", "none", "null", "<na>"})


def load_review_flag_hashes(review_flags_path: Path) -> set[str]:
    try:
        flags = pd.read_csv(review_flags_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ExerciseBankError(f"Unable to load exercise review flags from {review_flags_path}: {exc}") from exc
    required_flags = {"row_id", "problem_hash", "reason"}
    missing_flags = required_flags.difference(flags.columns)
    if missing_flags:
        raise ExerciseBankError(f"Review flags are missing required columns: {', '.join(sorted(missing_flags))}")
    if flags["problem_hash"].str.strip().eq("").any():
        raise ExerciseBankError("Review flags contain an empty problem_hash.")
    return set(flags["problem_hash"].str.strip())


def load_practice_bank(data: pd.DataFrame, review_flags_path: Path) -> pd.DataFrame:
    required_data = {"Problema", "Raspunsul", "problem_hash"}
    missing_data = required_data.difference(data.columns)
    if missing_data:
        raise ExerciseBankError(f"Exercise bank is missing required columns: {', '.join(sorted(missing_data))}")

    excluded_hashes = load_review_flag_hashes(review_flags_path)
    bank_hashes = set(data["problem_hash"].dropna().astype(str).str.strip())
    unmatched = excluded_hashes.difference(bank_hashes)
    if unmatched:
        raise ExerciseBankError(
            f"{len(unmatched)} review-flag problem_hash value(s) do not match the current bank. "
            "Inspect tools/check_generated_math.py and update flags only after content review."
        )
    hashes = data["problem_hash"].fillna("").astype(str).str.strip()
    keep = _has_content(data["Problema"]) & _has_content(data["Raspunsul"]) & ~hashes.isin(excluded_hashes)
    return data.loc[keep].copy()
