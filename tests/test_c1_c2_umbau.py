"""Tests for C1-Umbau: TLE invariance, C2 winner coupling, vLLM raw logprobs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent.compute_stages import VC_FOLLOWUP_PROMPT_MARKER, get_step_fn
from src.agent.stages.c2 import majority_vote
from src.signals import token_entropy
from src.utils.inference.vllm import _VLLM_LOGPROBS_MODE, VLLMWrapper


def _action_logprobs(text: str, action_lp: float = -0.2) -> list[dict]:
    """Build token logprobs aligned to text for action-line TLE slicing."""
    pieces = []
    if "</think>" in text:
        pre, post = text.split("</think>", 1)
        pieces.append(pre + "</think>")
        pieces.append(post)
    else:
        pieces = [text]
    out: list[dict] = []
    for i, chunk in enumerate(pieces):
        lp = -0.01 if i == 0 and len(pieces) > 1 else action_lp
        # Split chunk into coarse tokens for slicing tests.
        if chunk.strip() == "":
            continue
        for part in chunk.replace("\n", "\n|").split("|"):
            if part:
                out.append({"token": part, "logprob": lp})
    return out


def test_c0_c1_tle_same_position_for_matched_action_tokens():
    action_lp = -0.35
    c0_text = "go north\nextra"
    c1_text = "<think>\nplan\n</think>\ngo north\nextra"
    c0_lp = [
        {"token": "go", "logprob": action_lp},
        {"token": " north", "logprob": action_lp},
        {"token": "\n", "logprob": action_lp},
        {"token": "extra", "logprob": -5.0},
    ]
    c1_lp = [
        {"token": "<think>", "logprob": -0.01},
        {"token": "\n", "logprob": -0.01},
        {"token": "plan", "logprob": -0.01},
        {"token": "\n", "logprob": -0.01},
        {"token": "</think>", "logprob": -0.01},
        {"token": "\n", "logprob": -0.01},
        {"token": "go", "logprob": action_lp},
        {"token": " north", "logprob": action_lp},
        {"token": "\n", "logprob": action_lp},
        {"token": "extra", "logprob": -5.0},
    ]
    tle_c0 = token_entropy.extract_action_tle_from_response(c0_text, c0_lp)
    tle_c1 = token_entropy.extract_action_tle_from_response(c1_text, c1_lp)
    assert tle_c0 is not None and tle_c1 is not None
    assert tle_c0["mean_entropy"] == tle_c1["mean_entropy"]


def test_vllm_wrapper_pins_raw_logprobs_mode_on_engine():
    import sys

    wrapper = VLLMWrapper(model_name="Qwen/Qwen3-4B")
    assert wrapper.logprobs_mode == "raw_logprobs"
    assert _VLLM_LOGPROBS_MODE == "raw_logprobs"

    engine_captured: dict = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            engine_captured.update(kwargs)

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_tok = MagicMock()
    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok
    fake_vllm = MagicMock()
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = MagicMock()

    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers, "vllm": fake_vllm},
    ):
        with patch.object(VLLMWrapper, "_verify_logprob_invariance_capability"):
            wrapper._ensure_loaded()

    assert engine_captured.get("logprobs_mode") == "raw_logprobs"


def test_vllm_wrapper_sampling_params_omit_logprobs_mode():
    wrapper = VLLMWrapper(model_name="Qwen/Qwen3-4B")
    captured: dict = {}

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys

    fake_vllm = MagicMock()
    fake_vllm.SamplingParams = _FakeSamplingParams
    with patch.dict(sys.modules, {"vllm": fake_vllm}):
        wrapper._sampling_params(
            temperature=0.7,
            max_tokens=16,
            logprobs=True,
            merged_stop=None,
            extra={},
        )
    assert captured.get("logprobs") == 20
    assert "logprobs_mode" not in captured


def test_vllm_capability_probe_accepts_invariant_mock_responses():
    wrapper = VLLMWrapper(model_name="Qwen/Qwen3-4B")
    top = [
        {"token": "north", "logprob": -0.1},
        {"token": "south", "logprob": -2.0},
        {"token": "east", "logprob": -2.5},
    ]

    def _gen(_prompt, **kwargs):
        _ = kwargs
        return "north", [{"token": "north", "logprob": -0.1, "top_logprobs": list(top)}]

    wrapper.generate = _gen  # type: ignore[method-assign]
    wrapper._verify_logprob_invariance_capability()
    assert wrapper._logprob_invariance_verified is True


def test_c2_winner_tle_and_vc_use_same_winner_index_on_tie():
    """Forced tie: allocator TLE and VC followup both reference winner_index sample."""

    class _C2TieModel:
        def __init__(self) -> None:
            self.calls = 0
            self.vc_prompt: str | None = None

        def generate(self, prompt: str, logprobs: bool = False, **kwargs):
            self.calls += 1
            if VC_FOLLOWUP_PROMPT_MARKER in (prompt or ""):
                self.vc_prompt = prompt
                return "80", [{"logprob": -0.1}] * 2 if logprobs else None
            idx = (self.calls - 1) % 2
            if idx == 0:
                text = "<think>reason A</think>\ngo north"
            else:
                text = "<think>reason B</think>\ngo south"
            lp = _action_logprobs(text, action_lp=-0.25 - 0.1 * idx)
            return text, lp if logprobs else None

    m = _C2TieModel()
    step = get_step_fn(
        "C2",
        vc_mode="followup",
        c2_n_samples=2,
        c2_tie_break_seed="fixed-seed",
        c2_sample_temperature=0.7,
    )
    action, tle, vc, _tok, _calls, _lp, _vc_det, _prompt, _resp, call_detail = step("obs", [], m)
    assert action in {"go north", "go south"}
    assert tle is not None
    assert vc == 80.0
    assert m.vc_prompt is not None
    assert action in m.vc_prompt
    assert isinstance(call_detail, dict)
    wi = call_detail.get("winner_index")
    assert wi is not None
    winner_sc = next(
        sc for sc in call_detail.get("subcalls", []) if int(sc.get("sample_index", -1)) == int(wi)
    )
    assert winner_sc.get("is_winner") is True
    assert winner_sc.get("tle") == tle


def test_majority_vote_tie_break_is_deterministic():
    keys = ["a", "b"]
    rng1 = __import__("random").Random(42)
    rng2 = __import__("random").Random(42)
    w1, tb1, _ = majority_vote(keys, rng=rng1)
    w2, tb2, _ = majority_vote(keys, rng=rng2)
    assert tb1 and tb2
    assert w1 == w2
