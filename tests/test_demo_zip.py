import contextlib
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
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

    def test_nested_secrets_environments_symlinks_and_archives_are_excluded(self):
        (self.root / "app.py").write_text("application\n", encoding="utf-8")
        (self.root / "src").mkdir()
        (self.root / "src/required.py").write_text("required module\n", encoding="utf-8")
        (self.root / "data").mkdir()
        (self.root / "data/required.csv").write_text("required data\n", encoding="utf-8")
        (self.root / "models").mkdir()
        (self.root / "models/required.joblib").write_text("required model\n", encoding="utf-8")
        (self.root / "config").mkdir()
        (self.root / "config/.env").write_text("SYNTHETIC_ROOT_SECRET\n", encoding="utf-8")
        (self.root / "config/.env.production").write_text("SYNTHETIC_LIVE_ENV\n", encoding="utf-8")
        (self.root / "config/secrets.toml").write_text("SYNTHETIC_NESTED_SECRET\n", encoding="utf-8")
        (self.root / "config/id_rsa").write_text("SYNTHETIC_PRIVATE_KEY\n", encoding="utf-8")
        (self.root / "config/server_privatekey.p8").write_text("SYNTHETIC_PKCS8_KEY\n", encoding="utf-8")
        (self.root / "config/example.secret").write_text("SYNTHETIC_SECRET_SUFFIX\n", encoding="utf-8")
        (self.root / ".env.example").write_text("EXAMPLE_VALUE=allowed\n", encoding="utf-8")
        (self.root / "config/secrets.toml.example").write_text("placeholder=true\n", encoding="utf-8")

        for environment in (".venv-audit", "custom-python-env"):
            directory = self.root / environment
            directory.mkdir()
            (directory / "pyvenv.cfg").write_text("home = /synthetic\n", encoding="utf-8")
            (directory / "environment_payload.py").write_text("SYNTHETIC_ENV_PAYLOAD\n", encoding="utf-8")
        (self.root / "tools/__pycache__").mkdir(parents=True)
        (self.root / "tools/__pycache__/cached.pyc").write_bytes(b"SYNTHETIC_CACHE")
        (self.root / "generated.zip").write_bytes(b"SYNTHETIC_ARCHIVE")
        (self.root / "package.whl").write_bytes(b"SYNTHETIC_WHEEL")
        (self.root / "package.tar.gz").write_bytes(b"SYNTHETIC_TAR")

        internal_target = self.root / "data/internal-target.txt"
        internal_target.write_text("INTERNAL_TARGET_CONTENT\n", encoding="utf-8")
        internal_link = self.root / "data/internal-alias.txt"
        internal_link.symlink_to(internal_target)
        with tempfile.TemporaryDirectory() as outside_directory:
            outside_target = Path(outside_directory) / "outside.txt"
            outside_target.write_text("SYNTHETIC_OUTSIDE_SYMLINK_TARGET\n", encoding="utf-8")
            external_link = self.root / "data/outside-alias.txt"
            external_link.symlink_to(outside_target)
            outside_directory_link = self.root / "linked-directory"
            outside_directory_link.symlink_to(outside_directory, target_is_directory=True)

            original_read_bytes = Path.read_bytes
            opened_paths = []

            def tracked_read_bytes(path):
                opened_paths.append(path)
                return original_read_bytes(path)

            with patch.object(Path, "read_bytes", tracked_read_bytes):
                packaged, _ = self._build_and_check()

            self.assertNotIn(internal_link, opened_paths)
            self.assertNotIn(external_link, opened_paths)
            self.assertNotIn(outside_target, opened_paths)
            self.assertNotIn(outside_directory_link / "outside.txt", packaged)
            self.assertNotIn("data/internal-alias.txt", packaged)
            self.assertNotIn("data/outside-alias.txt", packaged)

        required = {
            "app.py", "src/required.py", "data/required.csv", "models/required.joblib",
            ".env.example", "config/secrets.toml.example", "data/internal-target.txt",
        }
        self.assertTrue(required.issubset(packaged))
        with ZipFile(build_demo_zip.OUTPUT) as archive:
            payload = b"\n".join(
                archive.read(name)
                for name in archive.namelist()
                if name.endswith("MANIFEST.sha256") or name != f"{build_demo_zip.ARCHIVE_ROOT}/MANIFEST.sha256"
            )
        for marker in (
            b"SYNTHETIC_ROOT_SECRET", b"SYNTHETIC_LIVE_ENV", b"SYNTHETIC_NESTED_SECRET",
            b"SYNTHETIC_PRIVATE_KEY", b"SYNTHETIC_SECRET_SUFFIX", b"SYNTHETIC_ENV_PAYLOAD",
            b"SYNTHETIC_PKCS8_KEY",
            b"SYNTHETIC_CACHE", b"SYNTHETIC_ARCHIVE", b"SYNTHETIC_WHEEL",
            b"SYNTHETIC_TAR", b"SYNTHETIC_OUTSIDE_SYMLINK_TARGET",
        ):
            self.assertNotIn(marker, payload)

    def test_docker_context_ignores_runtime_secrets_but_keeps_application_paths(self):
        repository = Path(__file__).resolve().parents[1]
        ignore_rules = (repository / ".dockerignore").read_text(encoding="utf-8")
        dockerfile = (repository / "Dockerfile").read_text(encoding="utf-8")
        launcher = (repository / "run_docker.sh").read_text(encoding="utf-8")

        for rule in (
            "**/.env", "**/.env.*", "**/secrets.toml", "**/id_rsa", "**/*.pem",
            "**/*.key", "**/*.secret", "**/*.zip", "**/__pycache__/",
            "**/.venv-*", "**/pyvenv.cfg",
        ):
            self.assertIn(rule, ignore_rules)
        for instruction in (
            "COPY app.py ./", "COPY src/ ./src/", "COPY data/ ./data/",
            "COPY models/ ./models/", "COPY .streamlit/ ./.streamlit/",
        ):
            self.assertIn(instruction, dockerfile)
        self.assertNotIn("COPY . /app", dockerfile)
        self.assertIn("docker run --rm -p 8501:8501", launcher)
        self.assertNotIn('-v "$(pwd)"', launcher)


if __name__ == "__main__":
    unittest.main()
