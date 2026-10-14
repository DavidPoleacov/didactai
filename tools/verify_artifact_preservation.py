#!/usr/bin/env python3
"""Compare preserved input artifacts with hashes recorded before the repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "750e8a424f8a33a27b1c9220fc262c31b3aba0df"
BASELINE_HASHES = {
    "data/raw/Exercises_augmented.xlsx": "4fbc9dea6f50c6a7104f4a670040ebb0029ca4df8447339c81d7b7e0d099c8a2",
    "data/raw/ai_judging_history.xlsx": "e1f7810328384d2791f67ea511cf2d74594bb7cc1a2883e66c75f3a4edf84982",
    "data/raw/domenii_categorii.xlsx": "9ac2baef9339d21af7d282d5a500929308a71659f97996f73db736dc4b64155a",
    "data/raw/exercises_corrected.xlsx": "4784314f0b4b82d4649c0ae0bda3013b6c223d458d99148b40dfdb8c3a24d178",
    "data/raw/jurizare_project_7.xlsx": "2ff6bfc8bba1f6a49a4316117b08ae9aa790312991296b153e34c57850a36c20",
    "data/processed/exercises_augmented.csv": "5e4d71d7cf583d7dffe58fd2828355af417ef19adbd004a0317a5d5e01b12f8d",
    "data/processed/exercises_processed.csv": "c7930a297be7de7f379662b643c3436304aea254ce5c34b78058a8bf784df273",
    "data/processed/student_interactions.csv": "b91ebe98425ba80f3e83aee4e7445cc0bdac3c5107cb8c13c58e04ebe46417d3",
    "models/evaluation_report.json": "144320c79f1db0846779c828486be0ffa9070d51fb949934080291ec0bf138b0",
    "models/feature_schema.json": "4ff74b7601ef4504ced1376973aba995295c34511234b1472eec9769e665acee",
    "models/neural_student_state_label_encoder.joblib": "742c3e425dfae0ea6eec94500e6178edf62982e9ee62498ed74470ed9c282146",
    "models/neural_student_state_model.keras": "13fa83aab223ad2b8e6374784f182ff2ab50c35f97226804cbd3938b832b5aee",
    "models/neural_student_state_report.json": "7bb873d39bcf25ec86afc8c1412b3a3e309c6d36092ada891eb9f71038e1bad6",
    "models/neural_student_state_scaler.joblib": "4bcca6155118018fe574caf03ad20c06e89cfd3dfe0ca9e95b2c19c15fe5ef76",
    "models/structured_difficulty_model.joblib": "f9b40e080fcf6ed1be836a77e3aefbf93b69d52d05cfbdcfbe5d38776181826b",
    "models/unstructured_domain_model.joblib": "de07663763f816bfd8b6e9f1004bf44b780bba8a3357952ed2488d39efc96b67",
}


def main() -> int:
    records = []
    mismatches = []
    for relative_path, before_hash in BASELINE_HASHES.items():
        path = ROOT / relative_path
        after_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        records.append({
            "path": relative_path,
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "unchanged": after_hash == before_hash,
        })
        if after_hash != before_hash:
            mismatches.append(relative_path)
    output = {
        "algorithm": "SHA-256",
        "baseline": f"pre-upload commit {BASELINE_COMMIT}",
        "files": records,
        "all_unchanged": not mismatches,
    }
    manifest_path = ROOT / "docs/artifact_preservation.json"
    manifest_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if mismatches:
        print("CHANGED PROTECTED ARTIFACTS: " + ", ".join(mismatches))
        return 1
    print(f"PASS: {len(records)} protected raw, processed, model, and report files are byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
