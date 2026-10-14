#!/usr/bin/env python3
"""Build a self-contained source ZIP with an inventory/hash manifest."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "DidactAI_Demo_Ready.zip"
ARCHIVE_ROOT = "DidactAI_Demo_Ready"
EXCLUDED_DIRS = {
    ".git", ".venv", ".venv-demo", ".venv-package-test", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist",
}
EXCLUDED_FILES = {".streamlit/secrets.toml", ".env", "secrets.toml", "id_rsa", "id_ed25519"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip", ".whl", ".tar.gz", ".pem", ".key"}


def source_files() -> list[Path]:
    selected = []
    for current, directories, filenames in os.walk(ROOT):
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_DIRS)
        current_path = Path(current)
        for filename in sorted(filenames):
            path = current_path / filename
            relative = path.relative_to(ROOT).as_posix()
            is_env_file = filename.startswith(".env.") and filename not in {
                ".env.example", ".env.template",
            }
            if (
                relative in EXCLUDED_FILES
                or relative == "MANIFEST.sha256"
                or is_env_file
                or any(relative.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)
            ):
                continue
            if filename.endswith(".secret"):
                continue
            if filename in {".DS_Store"}:
                continue
            selected.append(path)
    return sorted(selected, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> int:
    files = source_files()
    manifest_lines = [
        "# SHA-256 hashes for packaged source files.",
        "# MANIFEST.sha256 is intentionally excluded from its own hash list.",
    ]
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, f"{ARCHIVE_ROOT}/{path.relative_to(ROOT).as_posix()}")
        archive.writestr(f"{ARCHIVE_ROOT}/MANIFEST.sha256", manifest)
    print(f"Created {OUTPUT} with {len(files)} manifest-listed files plus MANIFEST.sha256.")
    print(f"Manifest sha256: {hashlib.sha256(manifest).hexdigest()}")
    print("Excluded: .git, virtual environments, Python caches, secrets, and nested archives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
