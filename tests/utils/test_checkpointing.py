"""Episode checkpoint listing ignores non-episode JSON."""

from __future__ import annotations

import json

import pytest

from src.utils.checkpointing import list_completed_episodes, save_episode_checkpoint


def test_list_completed_only_ep_prefix(tmp_path):
    (tmp_path / "ep_textworld_0_C0_0.json").write_text(
        json.dumps({"episode_id": "ep_textworld_0_C0_0"})
    )
    (tmp_path / "run_metadata.json").write_text("{}")
    (tmp_path / "run_info.json").write_text("{}")
    (tmp_path / "run_summary.json").write_text("{}")
    done = list_completed_episodes(tmp_path)
    assert done == {"ep_textworld_0_C0_0"}


def test_save_episode_checkpoint_leaves_no_partial_file_on_write_failure(tmp_path, monkeypatch):
    """Regression: log_episode() used to open(path, "w") + json.dump() directly onto the final
    ep_*.json path. A process killed mid-write left a truncated file that list_completed_episodes()
    -- a plain existence glob, not a content check -- then treated as done forever, permanently
    skipping that episode on --resume. Fixed via temp-file + os.replace (atomic on POSIX/Windows).
    See docs/consistency_log.md, 2026-07-20 Gate F entry; scripts/gate_f_resume_smoke.py reproduces
    the real hard-kill case end-to-end."""
    import src.utils.logging_utils as logging_utils

    real_dump = json.dump

    def _dump_then_crash(obj, fp, **kwargs):
        fp.write('{"task_success"')  # partial content, as a real mid-write kill would leave
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(logging_utils.json, "dump", _dump_then_crash)
    with pytest.raises(RuntimeError):
        save_episode_checkpoint(tmp_path, "ep_textworld_0_C0_0", {"task_success": True})

    final_path = tmp_path / "ep_textworld_0_C0_0.json"
    assert not final_path.exists(), (
        "a failed write must never leave a partial file at the final path"
    )
    assert list_completed_episodes(tmp_path) == set(), (
        "resume must not count a failed write as done"
    )
    assert list(tmp_path.glob("*.tmp")) == [], "temp file must not survive a crashed write"

    monkeypatch.setattr(logging_utils.json, "dump", real_dump)
    save_episode_checkpoint(tmp_path, "ep_textworld_0_C0_0", {"task_success": True})
    assert json.loads(final_path.read_text())["task_success"] is True
    assert list_completed_episodes(tmp_path) == {"ep_textworld_0_C0_0"}
