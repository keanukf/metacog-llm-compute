"""Episode checkpoint listing ignores non-episode JSON."""

from __future__ import annotations

import json

from src.utils.checkpointing import list_completed_episodes


def test_list_completed_only_ep_prefix(tmp_path):
    (tmp_path / "ep_textworld_0_C0_0.json").write_text(
        json.dumps({"episode_id": "ep_textworld_0_C0_0"})
    )
    (tmp_path / "run_metadata.json").write_text("{}")
    (tmp_path / "run_info.json").write_text("{}")
    (tmp_path / "run_summary.json").write_text("{}")
    done = list_completed_episodes(tmp_path)
    assert done == {"ep_textworld_0_C0_0"}
