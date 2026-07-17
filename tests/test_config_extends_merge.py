"""
`extends:` overlay merge for run_phase1.py / run_phase2.py (`_load_merged_config`).

Gate E rehearsal (2026-07-17, see docs/gate_e_rehearsal.md and docs/consistency_log.md) found
that `run_phase1.py`/`run_phase2.py::load_config` ignored `extends:` entirely (plain
`yaml.safe_load`, no merge) and fixed it by delegating to
`scripts.sweep_textworld_difficulty._load_merged_config`. That function existed already (used by
the Gate D diagnostic scripts) but had no direct test coverage, and its merge was only one level
deep: for a key whose value is itself a nested dict of nested dicts (e.g.
`domain_prompts.textworld.{prefix,cot_max_tokens,...}`), overriding a single leaf under
`domain_prompts.textworld` replaced the whole `textworld` sub-dict, silently dropping sibling keys
such as `prefix` and `action_stop`. No existing `configs/dev/*.yaml` overlay happened to override a
key at that depth, so the bug was latent (same shape as the two bugs Gate E did catch), not yet
observed in a live run.
"""

from __future__ import annotations

from pathlib import Path

from scripts.sweep_textworld_difficulty import _load_merged_config


def test_extends_merges_top_level_and_one_level_dicts(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "model:\n  name: base-model\n  dtype: float16\nepisode:\n  max_steps_per_episode: 25\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "extends: base.yaml\nlogging:\n  logprob_sidecar_mode: off\nepisode:\n"
        "  max_steps_per_episode: 6\n",
        encoding="utf-8",
    )
    merged = _load_merged_config(overlay)
    assert "extends" not in merged
    assert merged["model"] == {"name": "base-model", "dtype": "float16"}
    assert merged["episode"]["max_steps_per_episode"] == 6
    assert merged["logging"]["logprob_sidecar_mode"] is False


def test_extends_merge_preserves_sibling_keys_two_levels_deep(tmp_path: Path) -> None:
    """
    Regression for the shallow-merge bug: overriding domain_prompts.textworld.cot_max_tokens
    alone must not drop domain_prompts.textworld.prefix / action_stop, and must leave
    domain_prompts.tower_of_hanoi (untouched by the overlay) intact.
    """
    base = tmp_path / "base.yaml"
    base.write_text(
        "domain_prompts:\n"
        "  textworld:\n"
        "    prefix: BASE PREFIX\n"
        '    action_stop: ["\\n"]\n'
        "    cot_max_tokens: 999\n"
        "  tower_of_hanoi:\n"
        "    prefix: TOH BASE\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "extends: base.yaml\ndomain_prompts:\n  textworld:\n    cot_max_tokens: 8192\n",
        encoding="utf-8",
    )
    merged = _load_merged_config(overlay)
    tw = merged["domain_prompts"]["textworld"]
    assert tw["cot_max_tokens"] == 8192
    assert tw["prefix"] == "BASE PREFIX"
    assert tw["action_stop"] == ["\n"]
    assert merged["domain_prompts"]["tower_of_hanoi"] == {"prefix": "TOH BASE"}


def test_extends_chain_two_levels(tmp_path: Path) -> None:
    grandparent = tmp_path / "core.yaml"
    grandparent.write_text(
        "model:\n  name: core-model\ninference:\n  temperature: 0.3\n",
        encoding="utf-8",
    )
    parent = tmp_path / "mid.yaml"
    parent.write_text(
        "extends: core.yaml\ninference:\n  temperature: 0.5\n",
        encoding="utf-8",
    )
    child = tmp_path / "leaf.yaml"
    child.write_text(
        "extends: mid.yaml\nepisode:\n  max_steps_per_episode: 10\n",
        encoding="utf-8",
    )
    merged = _load_merged_config(child)
    assert merged["model"]["name"] == "core-model"
    assert merged["inference"]["temperature"] == 0.5
    assert merged["episode"]["max_steps_per_episode"] == 10


def test_no_extends_returns_raw_config(tmp_path: Path) -> None:
    cfg = tmp_path / "standalone.yaml"
    cfg.write_text("model:\n  name: standalone\n", encoding="utf-8")
    merged = _load_merged_config(cfg)
    assert merged == {"model": {"name": "standalone"}}
