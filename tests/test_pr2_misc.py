"""PR2: random allocation reproducibility and VC judged context."""

from __future__ import annotations

import hashlib

from src.agent.allocator import allocate
from src.agent.stages.shared import _build_model_output_to_judge_section, _build_vc_followup_prompt


def test_random_allocate_reproducible_by_episode_seed():
    ep_id = "ep_textworld_3_random_2"
    seed = int(hashlib.md5(ep_id.encode()).hexdigest()[:8], 16)
    import random

    rng1 = random.Random(seed)
    rng2 = random.Random(seed)
    seq1 = [allocate(None, "random", i, rng1) for i in range(8)]
    seq2 = [allocate(None, "random", i, rng2) for i in range(8)]
    assert seq1 == seq2


def test_vc_judged_context_action_only():
    block = _build_model_output_to_judge_section(
        "C1",
        "go north",
        judged_context="action_only",
        raw_action_completion="long cot...",
        cot_text="should not appear",
        verify_completion=None,
        c2_n_samples=None,
        c2_sample_first_lines=None,
        followup_cot_max_chars=100,
        raw_completion_max_chars=100,
    )
    assert block == "[C1] go north"
    assert "cot" not in block.lower()


def test_vc_followup_prompt_contains_action_only_block():
    prompt = _build_vc_followup_prompt(
        "obs",
        [],
        "prefix",
        stage_tag="C2",
        action_line="A->C",
        instruction="Confidence:",
        judged_context="action_only",
    )
    assert "<output_to_judge>" in prompt
    assert "[C2] A->C" in prompt
