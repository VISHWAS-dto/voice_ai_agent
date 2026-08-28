"""Tests for the API response schemas.

Stub only. Fill in once behavior stabilizes. Intended coverage:
- ``AnalyzeResponse`` round-trips the example payload from the contract.
- Confidence fields reject values outside [0, 1].
- Enum fields reject unknown strings.
"""

import pytest


@pytest.mark.skip(reason="schema tests not written yet")
def test_analyze_response_roundtrip() -> None:
    raise NotImplementedError
