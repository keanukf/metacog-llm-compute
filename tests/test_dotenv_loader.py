"""Repo-root .env loading (LM Studio, Langfuse)."""
from __future__ import annotations

import os

from src.utils.dotenv_loader import load_dotenv_if_present


def test_load_dotenv_override_true_replaces_empty_env_var(tmp_path, monkeypatch) -> None:
    """Empty pre-set env vars must not block .env (common with IDE/shell placeholders)."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".env").write_text("METACOG_DOTENV_TEST_X=from_file\n", encoding="utf-8")
    monkeypatch.setenv("METACOG_DOTENV_TEST_X", "")
    info = load_dotenv_if_present(root, override=True)
    assert info["loaded"] is True
    assert os.environ["METACOG_DOTENV_TEST_X"] == "from_file"


def test_load_dotenv_utf8_sig_strips_bom_on_first_key(tmp_path, monkeypatch) -> None:
    """Editors that save .env with UTF-8 BOM must not produce a bogus \\ufeff-prefixed env key name."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".env").write_bytes(b"\xef\xbb\xbfMETACOG_DOTENV_BOM=ok\n")
    monkeypatch.delenv("METACOG_DOTENV_BOM", raising=False)
    info = load_dotenv_if_present(root, override=True)
    assert info["loaded"] is True
    assert os.environ.get("METACOG_DOTENV_BOM") == "ok"
