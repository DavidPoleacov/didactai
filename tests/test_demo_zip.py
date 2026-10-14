import contextlib
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from tools import build_demo_zip


class DemoZipTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.previous_root = build_demo_zip.ROOT
        self.previous_output = build_demo_zip.OUTPUT
        self.addCleanup(self._restore_builder_paths)
        build_demo_zip.ROOT = self.root
        build_demo_zip.OUTPUT = self.root / "DidactAI_Demo_Ready.zip"

    def _restore_builder_paths(self):
        build_demo_zip.ROOT = self.previous_root
        build_demo_zip.OUTPUT = self.previous_output

    def _verify_archive(self, archive_path, root, expected_files=None):
        with ZipFile(archive_path) as archive:
            members = archive.namelist()
            self.assertEqual(len(members), len(set(members)), "ZIP member paths must be unique")
            prefix = f"{build_demo_zip.ARCHIVE_ROOT}/"
            packaged = {name[len(prefix):] for name in members if name.startswith(prefix)}
            manifest_path = "MANIFEST.sha256"
            self.assertIn(manifest_path, packaged)
            manifest_bytes = archive.read(f"{build_demo_zip.ARCHIVE_ROOT}/{manifest_path}")
            manifest_entries = {
                line.split("  ", 1)[1]: line.split("  ", 1)[0]
                for line in manifest_bytes.decode("utf-8").splitlines()
                if line and not line.startswith("#")
            }
            self.assertNotIn(manifest_path, manifest_entries)
            self.assertEqual(packaged, set(manifest_entries) | {manifest_path})
            if expected_files is not None:
                self.assertEqual(set(manifest_entries), expected_files)
            for relative_path, expected_hash in manifest_entries.items():
                content = archive.read(f"{build_demo_zip.ARCHIVE_ROOT}/{relative_path}")
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected_hash, relative_path)
            return set(manifest_entries), members.count(
                f"{build_demo_zip.ARCHIVE_ROOT}/{manifest_path}"
            )

    def _build_and_check(self, expected_files=None):
        with contextlib.redirect_stdout(io.StringIO()):
            build_demo_zip.main()
        return self._verify_archive(build_demo_zip.OUTPUT, self.root, expected_files)

    def test_clean_source_tree_has_complete_fresh_manifest(self):
        (self.root / "app.py").write_text("print('demo')\n", encoding="utf-8")
        (self.root / "README.md").write_text("Demo\n", encoding="utf-8")
        expected = {path.relative_to(self.root).as_posix() for path in build_demo_zip.source_files()}

        packaged, manifest_count = self._build_and_check(expected)

        self.assertEqual(packaged, expected)
        self.assertEqual(manifest_count, 1)

    def test_previous_manifest_is_not_packaged_or_hashed(self):
        (self.root / "app.py").write_text("print('new')\n", encoding="utf-8")
        (self.root / "MANIFEST.sha256").write_text("old digest  stale-file.py\n", encoding="utf-8")
        expected = {path.relative_to(self.root).as_posix() for path in build_demo_zip.source_files()}
        self.assertNotIn("MANIFEST.sha256", expected)

        packaged, manifest_count = self._build_and_check(expected)

        self.assertEqual(packaged, expected)
        self.assertEqual(manifest_count, 1)

    def test_rebuild_from_extracted_package_succeeds_without_duplicate_entries(self):
        repository = Path(__file__).resolve().parents[1]
        (self.root / "app.py").write_text("print('repackage')\n", encoding="utf-8")
        tools_directory = self.root / "tools"
        tools_directory.mkdir()
        (tools_directory / "build_demo_zip.py").write_bytes(
            (repository / "tools/build_demo_zip.py").read_bytes()
        )
        self._build_and_check()

        extracted = self.root / "extracted"
        with ZipFile(build_demo_zip.OUTPUT) as archive:
            archive.extractall(extracted)
        package_root = extracted / build_demo_zip.ARCHIVE_ROOT
        repackage = package_root / "tools/build_demo_zip.py"

        result = subprocess.run(
            [sys.executable, str(repackage)],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertNotIn("Duplicate name", result.stderr)
        second_archive = package_root / "DidactAI_Demo_Ready.zip"
        packaged, manifest_count = self._verify_archive(second_archive, package_root)
        self.assertEqual(manifest_count, 1)
        self.assertIn("app.py", packaged)

    def test_existing_secret_cache_and_nested_archive_exclusions_remain(self):
        (self.root / "app.py").write_text("print('safe')\n", encoding="utf-8")
        (self.root / ".env").write_text("TOKEN=not-real\n", encoding="utf-8")
        (self.root / ".streamlit").mkdir()
        (self.root / ".streamlit/secrets.toml").write_text("secret='not-real'\n", encoding="utf-8")
        (self.root / ".venv").mkdir()
        (self.root / ".venv/ignored.py").write_text("ignored\n", encoding="utf-8")
        (self.root / "nested.zip").write_bytes(b"not an archive")

        packaged, _ = self._build_and_check()

        self.assertEqual(packaged, {"app.py"})


if __name__ == "__main__":
    unittest.main()
