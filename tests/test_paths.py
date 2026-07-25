from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from src.infrastructure.paths import (
    PathSecurityError,
    atomic_write_bytes,
    atomic_write_text,
    ensure_directory,
    safe_join,
    sanitize_name,
)


class TestSanitizeName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("my drawing", "my-drawing"),
            ("../../etc/passwd", "etc-passwd"),
            ("a/b\\c", "a-b-c"),
            ("hello$(rm -rf /)", "hello-rm-rf"),
            ("session 2026-07-22", "session-2026-07-22"),
            ("..", "session"),
            ("", "session"),
            ("...", "session"),
            ("   ", "session"),
        ],
    )
    def test_reduces_to_a_safe_component(self, raw: str, expected: str) -> None:
        assert sanitize_name(raw) == expected

    def test_truncates_long_names(self) -> None:
        assert len(sanitize_name("x" * 500)) <= 96

    def test_result_is_always_a_single_component(self) -> None:
        for raw in ("../../x", "a/b/c", "C:\\Windows\\system32", "~/.ssh/id_rsa"):
            assert os.sep not in sanitize_name(raw)


class TestSafeJoin:
    def test_joins_inside_root(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "session-1", "drawing.png").parent.name == "session-1"

    def test_traversal_is_neutralised(self, tmp_path: Path) -> None:
        result = safe_join(tmp_path, "../../../../etc", "passwd")
        assert tmp_path.resolve() in result.parents

    def test_absolute_path_is_neutralised(self, tmp_path: Path) -> None:
        result = safe_join(tmp_path, "/etc/shadow")
        assert tmp_path.resolve() in result.parents

    def test_symlink_escape_is_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside-root"
        outside.mkdir(exist_ok=True)
        root = tmp_path / "root"
        root.mkdir()
        (root / "escape").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathSecurityError):
            safe_join(root, "escape", "loot.png")

    def test_requires_a_component(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            safe_join(tmp_path)


class TestAtomicWrites:
    def test_writes_payload(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "file.bin"
        atomic_write_bytes(target, b"payload")
        assert target.read_bytes() == b"payload"

    def test_overwrites_atomically(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        atomic_write_text(target, "first")
        atomic_write_text(target, "second")
        assert target.read_text() == "second"

    def test_leaves_no_temporary_files(self, tmp_path: Path) -> None:
        atomic_write_text(tmp_path / "a.json", "{}")
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")] == []

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_artifacts_are_owner_only(self, tmp_path: Path) -> None:
        target = atomic_write_text(tmp_path / "secret.json", "{}")
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_directories_are_owner_only(self, tmp_path: Path) -> None:
        directory = ensure_directory(tmp_path / "session")
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        
        
        
            #     *** _ ***