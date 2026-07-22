"""Tests the --real guard that stops a real run from silently using the mock model.

``_assert_real_model_or_raise`` must permit mock mode with no model but hard-fail (BackendError) when
``--real`` is set without a loaded model, so a production data-collection run can never quietly emit
mock outputs.
"""

from __future__ import annotations

import pytest

from scripts.experiment.run_pilot import _assert_real_model_or_raise
from src.utils.errors import BackendError


def test_assert_real_model_or_raise_allows_mock_mode_without_model() -> None:
    _assert_real_model_or_raise("mock", None, "unused")


def test_assert_real_model_or_raise_raises_backend_error_for_real_mode() -> None:
    with pytest.raises(BackendError, match="pilot_mode=cuda requested"):
        _assert_real_model_or_raise("cuda", None, "wrapper failed")
