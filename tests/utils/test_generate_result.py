"""``GenerateResult``: backward-compatible carrier for a backend's ``generate()`` return.

Verifies it unpacks/indexes/lens exactly like a bare ``(text, logprobs)`` 2-tuple -- the frozen
contract every stage call site and dozens of test mocks rely on -- while additionally exposing
``.prompt_tokens`` for call sites that opt in. This is what makes prompt-token tracking (P1-stat-7)
purely additive: nothing that only expects the 2-tuple shape can observe a difference.
"""

from __future__ import annotations

from src.utils.inference.generate_result import GenerateResult


def test_unpacks_like_a_two_tuple():
    r = GenerateResult("go north", [{"token": "go"}], prompt_tokens=42)
    text, logprobs = r
    assert text == "go north"
    assert logprobs == [{"token": "go"}]


def test_len_and_indexing_match_two_tuple_semantics():
    r = GenerateResult("go north", None, prompt_tokens=42)
    assert len(r) == 2
    assert r[0] == "go north"
    assert r[1] is None


def test_prompt_tokens_attribute_accessible_without_unpacking():
    r = GenerateResult("go north", None, prompt_tokens=42)
    assert r.prompt_tokens == 42


def test_getattr_on_a_bare_tuple_defaults_gracefully():
    """A mock returning a plain tuple (as every test mock in the suite does) has no
    ``prompt_tokens`` attribute -- call sites must fall back to 0/None via getattr, never raise."""
    bare = ("go north", None)
    assert getattr(bare, "prompt_tokens", None) is None
