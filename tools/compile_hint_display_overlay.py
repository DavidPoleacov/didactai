#!/usr/bin/env python3
"""Compile the reviewed hint-display overlay from the pinned exercise snapshot."""
from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/exercises_augmented.csv"
OUTPUT = ROOT / "data/processed/hint_display_corrections.csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pedagogical_engine import (  # noqa: E402
    split_solution_steps,
    validate_discriminant_display_correction,
)
EXPECTED_SOURCE_SHA256 = "5e4d71d7cf583d7dffe58fd2828355af417ef19adbd004a0317a5d5e01b12f8d"
REVIEWED_INDICES = {
    333, 337, 338, 339, 341, 342, 343, 344, 345, 347, 348, 349, 350, 357, 360,
    361, 367, 368, 369, 370, 371, 372, 374, 377, 378, 379, 380, 382, 384, 389,
    398, 400, 402, 404, 407, 409, 410, 411, 415, 420, 421, 423, 424, 427, 429,
    430, 432, 1116, 1121, 1126, 1144, 1146, 1148, 1152, 1155, 1159, 1161, 1168,
    1175, 1176,
}
RATIONALE = (
    "Parenthesize signed b before squaring; coefficients and reported discriminant "
    "checked with bounded integer arithmetic."
)
NEGATIVE_SQUARE = re.compile(r"(?:Δ|delta)[^;]*?-\s*\d+\s*(?:²|\^2)", re.IGNORECASE)


def main() -> int:
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "The protected exercise CSV differs from the reviewed snapshot; "
            "do not regenerate the hint overlay without renewed review."
        )

    data = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    rows = []
    discovered = set()
    for index, row in data.iterrows():
        for step_number, original_step in enumerate(
            split_solution_steps(row["Pasii de rezolvare"]), start=1,
        ):
            if not NEGATIVE_SQUARE.search(original_step):
                continue
            discovered.add(int(index))
            corrected_step = re.sub(
                r"(?<!\()(-\s*\d+)\s*(²|\^2)",
                r"(\1)\2",
                original_step,
                count=1,
            )
            if not validate_discriminant_display_correction(
                row["Problema"], original_step, corrected_step,
            ):
                raise ValueError(f"Discriminant correction failed review at stable index {index}.")
            rows.append({
                "stable_index": int(index),
                "problem_hash": row["problem_hash"],
                "step_number": step_number,
                "original_step": original_step,
                "corrected_step": corrected_step,
                "rationale": RATIONALE,
                "source_step_sha256": hashlib.sha256(original_step.encode("utf-8")).hexdigest(),
            })

    if discovered != REVIEWED_INDICES:
        missing = sorted(REVIEWED_INDICES.difference(discovered))
        unexpected = sorted(discovered.difference(REVIEWED_INDICES))
        raise ValueError(f"Reviewed hint candidate set changed; missing={missing}, unexpected={unexpected}.")

    columns = [
        "stable_index", "problem_hash", "step_number", "original_step",
        "corrected_step", "rationale", "source_step_sha256",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} reviewed display corrections to {OUTPUT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
