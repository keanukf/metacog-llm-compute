"""Live run monitor (``progress_watch.py``): directory-based snapshot, no run-script coupling.

Verifies domain/stage classification from production and QC-probe run-directory naming (with an
``unknown`` fallback), completed-vs-non-episode-JSON counting, in-flight trace-line counting that
excludes finished episodes, and the rendered expected-ratio breakdown. It reads only the output
directory, never the running process, so monitoring a long RunPod run cannot perturb or slow the
run it is watching -- the decoupling is the point.
"""

from __future__ import annotations

import json
import time

from scripts.run_readiness.progress_watch import _classify, render, snapshot


def test_classify_extracts_domain_and_stage_from_production_naming():
    assert _classify("ep_textworld_3_C1_0") == ("textworld", "C1")
    assert _classify("ep_tower_of_hanoi_12_C2_4") == ("tower_of_hanoi", "C2")


def test_classify_extracts_domain_and_stage_from_qc_probe_naming():
    assert _classify("qc_textworld_2_C2") == ("textworld", "C2")


def test_classify_falls_back_to_unknown_for_unrecognized_names():
    assert _classify("weird_file_name") == ("?", "?")


def test_snapshot_counts_completed_and_skips_non_episode_json(tmp_path):
    (tmp_path / "ep_textworld_0_C0_0.json").write_text(
        json.dumps({"task_success": True, "episode_length_steps": 12})
    )
    (tmp_path / "ep_textworld_1_C1_0.json").write_text(
        json.dumps({"task_success": False, "episode_length_steps": 8})
    )
    (tmp_path / "run_metadata.json").write_text("{}")  # must be ignored, not an episode file

    snap = snapshot(tmp_path)
    assert snap["done_total"] == 2
    assert snap["done"][("textworld", "C0")] == 1
    assert snap["done"][("textworld", "C1")] == 1
    assert snap["total_steps_observed"] == 20


def test_snapshot_counts_inflight_trace_lines_but_not_for_finished_episodes(tmp_path):
    # Finished episode: its trace file's lines must not double-count into in-flight.
    (tmp_path / "ep_textworld_0_C1_0.json").write_text(
        json.dumps({"task_success": True, "episode_length_steps": 3})
    )
    (tmp_path / "trace_ep_textworld_0_C1_0.jsonl").write_text(
        '{"step_index": 0}\n{"step_index": 1}\n'
    )
    # In-flight episode: no ep_*.json yet, trace lines count as in-flight progress.
    (tmp_path / "trace_ep_tower_of_hanoi_1_C2_0.jsonl").write_text(
        '{"step_index": 0}\n{"step_index": 1}\n{"step_index": 2}\n'
    )

    snap = snapshot(tmp_path)
    assert snap["done_total"] == 1
    assert snap["inflight"][("tower_of_hanoi", "C2")] == 1
    assert snap["inflight_steps_total"] == 3
    assert snap["total_steps_observed"] == 3 + 3  # finished episode's own steps + in-flight lines


def test_render_shows_expected_ratio_and_per_cell_breakdown():
    snap = {
        "done": {("textworld", "C1"): 2, ("tower_of_hanoi", "C2"): 1},
        "done_total": 3,
        "inflight": {("tower_of_hanoi", "C2"): 1},
        "inflight_steps_total": 5,
        "total_steps_observed": 50,
        "earliest_mtime": time.time() - 60,
    }
    out = render(snap, expected=8, elapsed_s=60.0)
    assert "3/8" in out
    assert "textworld/C1: 2 done" in out
    assert "tower_of_hanoi/C2: 1 done (+1 in flight)" in out
