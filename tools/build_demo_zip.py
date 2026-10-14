#!/usr/bin/env python3
"""Build a self-contained source ZIP with an inventory/hash manifest."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "DidactAI_Demo_Ready.zip"
ARCHIVE_ROOT = "DidactAI_Demo_Ready"
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".tox", ".nox",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    ".streamlit-cache", "node_modules", "build", "dist",
}
EXAMPLE_ENV_FILES = {".env.example", ".env.template", ".env.sample"}
EXAMPLE_SECRET_FILES = {
    "secrets.example.toml", "secrets.template.toml", "secrets.toml.example",
    "credentials.example.json", "credentials.template.json",
}
SENSITIVE_FILES = {
    "secrets.toml", "secrets.json", "credentials.json", ".netrc", ".npmrc",
    ".pypirc", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".zip", ".whl", ".tar.gz", ".pem", ".key", ".p8", ".ppk", ".p12", ".pfx",
    ".secret",
}


def _is_excluded_filename(filename: str) -> bool:
    lowered = filename.casefold()
    if lowered in EXAMPLE_ENV_FILES or lowered in EXAMPLE_SECRET_FILES:
        return False
    if lowered in SENSITIVE_FILES or lowered == "secrets.toml":
        return True
    if lowered == ".env" or (lowered.startswith(".env.") and lowered not in EXAMPLE_ENV_FILES):
        return True
    if (
        lowered.startswith("id_")
        or "private-key" in lowered
        or "private_key" in lowered
        or "privatekey" in lowered
    ):
        return True
    return any(lowered.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def _is_virtual_environment(directory: Path) -> bool:
    if directory.name.casefold() in EXCLUDED_DIRS:
        return True
    return os.path.lexists(directory / "pyvenv.cfg")


def _safe_regular_file(path: Path, root: Path) -> bool:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return False
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _safe_file_bytes(path: Path, root: Path) -> bytes | None:
    if not _safe_regular_file(path, root):
        return None
    return path.read_bytes()


def source_files() -> list[Path]:
    root = Path(ROOT).resolve()
    selected = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories
            if not (current_path / name).is_symlink()
            and not _is_virtual_environment(current_path / name)
        )
        for filename in sorted(filenames):
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            if (
                relative == "MANIFEST.sha256"
                or _is_excluded_filename(filename)
                or filename == ".DS_Store"
                or not _safe_regular_file(path, root)
            ):
                continue
            selected.append(path)
    return sorted(selected, key=lambda item: item.relative_to(root).as_posix())


def main() -> int:
    root = Path(ROOT).resolve()
    packaged = []
    for path in source_files():
        content = _safe_file_bytes(path, root)
        if content is not None:
            packaged.append((path, path.relative_to(root).as_posix(), content))

    manifest_lines = [
        "# SHA-256 hashes for packaged source files.",
        "# MANIFEST.sha256 is intentionally excluded from its own hash list.",
    ]
    for _, relative, content in packaged:
        manifest_lines.append(f"{hashlib.sha256(content).hexdigest()}  {relative}")
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for _, relative, content in packaged:
            archive.writestr(f"{ARCHIVE_ROOT}/{relative}", content)
        archive.writestr(f"{ARCHIVE_ROOT}/MANIFEST.sha256", manifest)
    print(f"Created {OUTPUT} with {len(packaged)} manifest-listed files plus MANIFEST.sha256.")
    print(f"Manifest sha256: {hashlib.sha256(manifest).hexdigest()}")
    print("Excluded: symlinks, virtual environments, caches, secrets, and nested archives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
