"""Unit tests for :func:`app.audio.quality.assess_quality`.

These pin the exact decision boundaries of the quality gate so a future
retune of the thresholds is a deliberate, visible change (every number in
``app/audio/quality.py`` is currently a hand-picked starting point, not
calibrated against labeled data — see the module docstring there).

The gate's contract, restated:

*   ``speech_ratio >= 0.5``                              -> ``"good"``
*   ``0.15 <= speech_ratio < 0.5``                       -> ``"degraded"``
*   ``speech_ratio < 0.15``                              -> ``"insufficient"``
*   ``total_duration_s < 0.5``  (any ratio)              -> ``"insufficient"``
*   ``speech_duration_s < 0.5`` (any ratio)              -> ``"insufficient"``

Boundary convention: ``0.15`` and ``0.5`` are **inclusive** lower edges of
the bucket they open (``>= 0.15`` is at least ``degraded``; ``>= 0.5`` is
``good``), because the implementation tests ``< threshold``.

Every test builds the ``vad_stats`` dict by hand — the real VAD is not
exercised here; :func:`assess_quality` only reads three float keys off it.
"""

from __future__ import annotations

import pytest

from app.audio.quality import assess_quality

GOOD = "good"
DEGRADED = "degraded"
INSUFFICIENT = "insufficient"


def _stats(
    speech_ratio: float,
    *,
    total_duration_s: float = 10.0,
    speech_duration_s: float | None = None,
    num_speech_segments: int = 3,
) -> dict:
    """Build a ``run_vad``-shaped stats dict.

    ``speech_duration_s`` defaults to ``speech_ratio * total_duration_s`` so
    that, unless a test overrides it, the two absolute-duration floors are
    satisfied whenever ``total_duration_s`` is comfortably above 0.5 s and
    ``speech_ratio`` is not tiny — keeping each test focused on the single
    knob it means to vary.
    """
    if speech_duration_s is None:
        speech_duration_s = speech_ratio * total_duration_s
    return {
        "total_duration_s": total_duration_s,
        "speech_duration_s": speech_duration_s,
        "speech_ratio": speech_ratio,
        "num_speech_segments": num_speech_segments,
    }


# --- the three ratio bands (well clear of the boundaries) -----------------


@pytest.mark.parametrize("ratio", [0.5, 0.6, 0.75, 0.9, 1.0])
def test_high_speech_ratio_is_good(ratio: float) -> None:
    """>= 0.5 of a long clip being speech -> 'good'."""
    assert assess_quality(_stats(ratio)) == GOOD


@pytest.mark.parametrize("ratio", [0.15, 0.2, 0.3, 0.45, 0.49])
def test_mid_speech_ratio_is_degraded(ratio: float) -> None:
    """Real but minority speech (0.15 <= ratio < 0.5) -> 'degraded'."""
    assert assess_quality(_stats(ratio)) == DEGRADED


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.05, 0.1, 0.149])
def test_low_speech_ratio_is_insufficient(ratio: float) -> None:
    """Near-silence (ratio < 0.15) -> 'insufficient', regardless of length."""
    assert assess_quality(_stats(ratio)) == INSUFFICIENT


# --- exact threshold boundaries ----------------------------------------
#
# The implementation branches on `< _MIN_SPEECH_RATIO_USABLE` (0.15) and
# `< _MIN_SPEECH_RATIO_GOOD` (0.5), so both constants are the *inclusive*
# bottom of the bucket they open. These tests would flip if either
# comparison were changed to `<=`.


def test_boundary_ratio_exactly_0_15_is_degraded_not_insufficient() -> None:
    """0.15 is the inclusive floor of 'degraded' (0.15 < 0.15 is False)."""
    assert assess_quality(_stats(0.15)) == DEGRADED


def test_boundary_just_below_0_15_is_insufficient() -> None:
    """A hair under 0.15 falls into 'insufficient'."""
    assert assess_quality(_stats(0.15 - 1e-9)) == INSUFFICIENT


def test_boundary_ratio_exactly_0_5_is_good_not_degraded() -> None:
    """0.5 is the inclusive floor of 'good' (0.5 < 0.5 is False)."""
    assert assess_quality(_stats(0.5)) == GOOD


def test_boundary_just_below_0_5_is_degraded() -> None:
    """A hair under 0.5 stays 'degraded'."""
    assert assess_quality(_stats(0.5 - 1e-9)) == DEGRADED


# --- total_duration_s floor overrides an otherwise-fine ratio ----------


def test_short_total_duration_is_insufficient_even_with_full_speech() -> None:
    """A 0.3 s clip that is 100% speech is still 'insufficient'."""
    stats = _stats(1.0, total_duration_s=0.3, speech_duration_s=0.3)
    assert assess_quality(stats) == INSUFFICIENT


def test_total_duration_exactly_0_5_with_good_ratio_is_good() -> None:
    """0.5 s is the inclusive floor (0.5 < 0.5 is False) -> ratio decides."""
    stats = _stats(1.0, total_duration_s=0.5, speech_duration_s=0.5)
    assert assess_quality(stats) == GOOD


def test_total_duration_just_below_0_5_is_insufficient() -> None:
    """Just under half a second of audio -> 'insufficient' whatever the ratio."""
    stats = _stats(1.0, total_duration_s=0.5 - 1e-9, speech_duration_s=0.5 - 1e-9)
    assert assess_quality(stats) == INSUFFICIENT


# --- speech_duration_s floor overrides an otherwise-fine ratio --------


def test_tiny_speech_duration_is_insufficient_even_with_good_ratio() -> None:
    """0.6 s clip, 60% speech -> ratio says 'good' but only 0.36 s of speech."""
    stats = _stats(0.6, total_duration_s=0.6, speech_duration_s=0.36)
    assert assess_quality(stats) == INSUFFICIENT


def test_speech_duration_exactly_0_5_is_not_the_thing_that_fails() -> None:
    """With >=0.5 s of speech and a good ratio on a long clip -> 'good'."""
    stats = _stats(0.6, total_duration_s=10.0, speech_duration_s=0.5)
    assert assess_quality(stats) == GOOD


# --- robustness of the input handling --------------------------------


def test_empty_stats_dict_degrades_to_insufficient() -> None:
    """Missing keys are read as 0.0 -> 'insufficient', never a KeyError."""
    assert assess_quality({}) == INSUFFICIENT


def test_none_valued_keys_are_treated_as_zero() -> None:
    """A stats dict with None values must not raise."""
    stats = {
        "total_duration_s": None,
        "speech_duration_s": None,
        "speech_ratio": None,
        "num_speech_segments": None,
    }
    assert assess_quality(stats) == INSUFFICIENT


def test_return_value_is_a_plain_string() -> None:
    """Callers compare against AudioQuality.*.value, so a str must come back."""
    result = assess_quality(_stats(0.9))
    assert isinstance(result, str)
    assert result in {GOOD, DEGRADED, INSUFFICIENT}
