"""Tests for base scanner functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from ai_bom.models import AIComponent
from ai_bom.scanners.base import (
    BaseScanner,
    _load_ignore_spec,
    _reset_ignore_spec,
    get_all_scanners,
)


class ConcreteScanner(BaseScanner):
    """Concrete scanner for testing base class functionality."""

    name = "test_scanner"
    description = "Test scanner"

    def supports(self, path: Path) -> bool:
        return True

    def scan(self, path: Path) -> list[AIComponent]:
        return []


@pytest.fixture
def scanner():
    return ConcreteScanner()


@pytest.fixture(autouse=True)
def reset_ignore():
    """Reset the global ignore spec before each test."""
    _reset_ignore_spec()
    yield
    _reset_ignore_spec()


class TestBaseScannerSafeReadText:
    def test_safe_read_text_utf8(self, scanner, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello UTF-8 ✓", encoding="utf-8")
        content = scanner.safe_read_text(test_file)
        assert content == "Hello UTF-8 ✓"

    def test_safe_read_text_binary_file(self, scanner, tmp_path):
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")
        content = scanner.safe_read_text(test_file)
        assert content is None

    def test_safe_read_text_latin1_fallback(self, scanner, tmp_path):
        test_file = tmp_path / "test.txt"
        # Write latin-1 encoded text
        test_file.write_bytes(b"Hello \xe9")  # é in latin-1
        content = scanner.safe_read_text(test_file)
        assert content is not None
        assert "Hello" in content

    def test_safe_read_text_permission_error(self, scanner, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            content = scanner.safe_read_text(test_file)
            assert content is None

    def test_safe_read_text_os_error(self, scanner, tmp_path):
        test_file = tmp_path / "nonexistent.txt"
        content = scanner.safe_read_text(test_file)
        assert content is None


class TestBaseScannerIterFiles:
    def test_iter_files_with_extension_filter(self, scanner, tmp_path):
        (tmp_path / "test.py").write_text("pass")
        (tmp_path / "test.js").write_text("pass")
        (tmp_path / "test.txt").write_text("pass")

        py_files = list(scanner.iter_files(tmp_path, extensions={".py"}))
        assert len(py_files) == 1
        assert py_files[0].name == "test.py"

    def test_iter_files_with_filename_filter(self, scanner, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.12")
        (tmp_path / "requirements.txt").write_text("openai")
        (tmp_path / "app.py").write_text("pass")

        dockerfiles = list(scanner.iter_files(tmp_path, filenames={"Dockerfile"}))
        assert len(dockerfiles) == 1
        assert dockerfiles[0].name == "Dockerfile"

    def test_iter_files_single_file_matching(self, scanner, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")

        files = list(scanner.iter_files(test_file, extensions={".py"}))
        assert len(files) == 1
        assert files[0] == test_file

    def test_iter_files_single_file_not_matching(self, scanner, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("pass")

        files = list(scanner.iter_files(test_file, extensions={".py"}))
        assert len(files) == 0

    def test_iter_files_excludes_pyc(self, scanner, tmp_path):
        (tmp_path / "test.py").write_text("pass")
        (tmp_path / "test.pyc").write_text("compiled")

        files = list(scanner.iter_files(tmp_path))
        assert all(f.suffix != ".pyc" for f in files)

    def test_iter_files_excludes_large_files(self, scanner, tmp_path):
        large_file = tmp_path / "large.txt"
        # Create a file larger than 10MB
        with patch("os.path.getsize", return_value=10_485_761):
            large_file.write_text("test")
            files = list(scanner.iter_files(tmp_path))
            assert large_file not in files

    def test_iter_files_excludes_binary(self, scanner, tmp_path):
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x00\x00\x00test")

        files = list(scanner.iter_files(tmp_path))
        assert binary_file not in files

    def test_iter_files_excludes_test_dirs(self, scanner, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text("pass")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_app.py").write_text("pass")

        files = list(scanner.iter_files(tmp_path, extensions={".py"}))
        assert all("tests" not in str(f) for f in files)

    def test_iter_files_includes_test_dirs_when_requested(self, scanner, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_app.py").write_text("pass")

        files = list(scanner.iter_files(tmp_path, extensions={".py"}, include_tests=True))
        assert any("tests" in str(f) for f in files)

    def test_iter_files_excludes_node_modules(self, scanner, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.js").write_text("pass")
        node_modules_dir = tmp_path / "node_modules" / "lib"
        node_modules_dir.mkdir(parents=True)
        (node_modules_dir / "index.js").write_text("pass")

        files = list(scanner.iter_files(tmp_path, extensions={".js"}))
        assert all("node_modules" not in str(f) for f in files)

    def test_iter_files_handles_permission_error(self, scanner, tmp_path):
        with patch("os.walk", side_effect=PermissionError("Access denied")):
            files = list(scanner.iter_files(tmp_path))
            assert files == []

    def test_iter_files_skips_pyc_single_file(self, scanner, tmp_path):
        # Test that .pyc files are skipped even when passed directly
        pyc_file = tmp_path / "test.pyc"
        pyc_file.write_text("compiled")

        files = list(scanner.iter_files(pyc_file))
        assert len(files) == 0

    def test_iter_files_no_filters_matches_all(self, scanner, tmp_path):
        (tmp_path / "file1.py").write_text("pass")
        (tmp_path / "file2.txt").write_text("pass")
        (tmp_path / "file3.md").write_text("pass")

        files = list(scanner.iter_files(tmp_path))
        assert len(files) == 3


class TestIgnoreSpec:
    def test_load_ignore_spec_not_found(self, tmp_path):
        spec = _load_ignore_spec(tmp_path)
        assert spec is None

    def test_load_ignore_spec_without_pathspec(self, tmp_path):
        ignore_file = tmp_path / ".ai-bomignore"
        ignore_file.write_text("*.pyc\nnode_modules/\n")

        with patch.dict("sys.modules", {"pathspec": None}):
            _reset_ignore_spec()
            spec = _load_ignore_spec(tmp_path)
            assert spec is None

    def test_load_ignore_spec_caching(self, tmp_path):
        ignore_file = tmp_path / ".ai-bomignore"
        ignore_file.write_text("*.pyc\n")

        try:
            import pathspec  # noqa: F401

            spec1 = _load_ignore_spec(tmp_path)
            spec2 = _load_ignore_spec(tmp_path)
            assert spec1 is spec2  # Should return cached spec
        except ImportError:
            pytest.skip("pathspec not installed")

    def test_iter_files_respects_ignore_spec(self, scanner, tmp_path):
        ignore_file = tmp_path / ".ai-bomignore"
        ignore_file.write_text("*.log\ntemp/\n")

        (tmp_path / "app.py").write_text("pass")
        (tmp_path / "debug.log").write_text("logs")
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        (temp_dir / "cache.txt").write_text("cache")

        try:
            import pathspec  # noqa: F401

            _reset_ignore_spec()
            files = list(scanner.iter_files(tmp_path))
            filenames = [f.name for f in files]

            assert "app.py" in filenames
            assert "debug.log" not in filenames
            assert "cache.txt" not in filenames
        except ImportError:
            pytest.skip("pathspec not installed")


class TestGetAllScanners:
    def test_get_all_scanners_returns_list(self):
        scanners = get_all_scanners()
        assert isinstance(scanners, list)
        assert len(scanners) > 0

    def test_get_all_scanners_contains_instances(self):
        scanners = get_all_scanners()
        for scanner in scanners:
            assert isinstance(scanner, BaseScanner)
            assert scanner.name != ""
            assert scanner.description != ""


class TestScannerRegistration:
    def test_scanner_auto_registration(self):
        class TestAutoScanner(BaseScanner):
            name = "auto_test"
            description = "Auto registered"

            def supports(self, path: Path) -> bool:
                return True

            def scan(self, path: Path) -> list[AIComponent]:
                return []

        scanners = get_all_scanners()
        scanner_names = [s.name for s in scanners]
        assert "auto_test" in scanner_names

    def test_scanner_without_name_not_registered(self):
        class AbstractTestScanner(BaseScanner):
            # No name attribute - should not be registered
            description = "Abstract scanner"

            def supports(self, path: Path) -> bool:
                return True

            def scan(self, path: Path) -> list[AIComponent]:
                return []

        scanners = get_all_scanners()
        # Should not have an empty name scanner
        assert all(s.name != "" for s in scanners)
