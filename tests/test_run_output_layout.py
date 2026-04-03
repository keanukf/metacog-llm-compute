"""Timestamped run directories."""
from __future__ import annotations

from pathlib import Path

from src.utils.run_output_layout import make_run_subdirectory, write_short_run_info


def test_make_run_subdirectory_creates_named_folder(tmp_path: Path):
    d = make_run_subdirectory(tmp_path, prefix="pilot")
    assert d.parent == tmp_path
    assert d.name.startswith("pilot_")
    assert d.is_dir()


def test_write_short_run_info(tmp_path: Path):
    p = write_short_run_info(
        tmp_path,
        script="t.py",
        config_path="/x/c.yaml",
        extra={"k": 1, "pth": tmp_path / "sub"},
    )
    assert p.name == "run_info.json"
    assert p.exists()
    import json

    body = json.loads(p.read_text())
    assert body["pth"] == str(tmp_path / "sub")
