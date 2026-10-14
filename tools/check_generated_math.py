#!/usr/bin/env python3
"""Audit dataset provenance and known limitation labels without evaluating expressions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CSV_SHA256 = "5e4d71d7cf583d7dffe58fd2828355af417ef19adbd004a0317a5d5e01b12f8d"


def audit() -> dict:
    source = pd.read_csv(ROOT / "data/processed/exercises_augmented.csv")
    flags = pd.read_csv(ROOT / "data/processed/exercise_review_flags.csv", dtype=str, keep_default_na=False)
    flagged = set(flags["problem_hash"])
    bank_hashes = set(source["problem_hash"].dropna().astype(str))
    source_types = source["Sursa_type"].fillna("unknown").value_counts().to_dict()
    excluded = source[source["problem_hash"].isin(flagged)]
    generated = source[source["Sursa_type"].astype(str).str.casefold().eq("alta")]
    actual_sha = hashlib.sha256((ROOT / "data/processed/exercises_augmented.csv").read_bytes()).hexdigest()
    unsupported = Counter()
    for reason in flags["reason"]:
        label = reason.split(":", 1)[0].strip()
        unsupported[label] += 1
    return {
        "source_rows": int(len(source)),
        "source_csv_sha256": actual_sha,
        "source_csv_matches_audited_snapshot": actual_sha == EXPECTED_CSV_SHA256,
        "source_type_counts": {str(key): int(value) for key, value in source_types.items()},
        "generated_rows": int(len(generated)),
        "source_derived_rows": int(len(source) - len(generated)),
        "review_flags": int(len(flags)),
        "matched_exclusions": int(len(excluded)),
        "unmatched_exclusion_hashes": sorted(flagged.difference(bank_hashes)),
        "available_practice_rows": int(len(source) - len(excluded)),
        "limitation_categories": dict(unsupported),
        "warning": (
            "This audit does not prove mathematical correctness. Expressions were not executed; "
            "unflagged prompts have not thereby received teacher review. Stored-answer round trips "
            "test formatting compatibility only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for a JSON copy of the report.")
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["matched_exclusions"] == report["review_flags"] and not report["unmatched_exclusion_hashes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
