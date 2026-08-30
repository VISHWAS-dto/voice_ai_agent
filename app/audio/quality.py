"""Audio quality gating for the inference pipeline.

Turns the coarse VAD stats from :func:`app.audio.ingest.run_vad` into the
three-way :class:`~app.schemas.models.AudioQuality` rating the API surfaces,
and — more importantly — into a go / no-go decision on whether it is worth
running the ~1 GB acoustic model on this clip at all.

The single signal we gate on is **speech ratio** (seconds of VAD-detected
speech / total clip seconds), with an absolute floor on how much speech
there is in wall-clock terms. We deliberately do *not* fold in SNR,
clipping, or level estimates here: the model we use
(``wav2vec2-large-robust``) was pre-trained on deliberately noisy /
telephone audio and tolerates a low SNR far better than it tolerates
*near-silence*, where it regresses toward a confident-looking but
meaningless answer (observed in ``scripts/test_inference.py`` on a silent
buffer). "Is there actually a voice here?" is the question that matters.
"""

from app.schemas.models import AudioQuality

# --- thresholds ------------------------------------------------------------
#
# These are a hand-picked STARTING POINT, not tuned against labeled data.
# They should be calibrated against a real set of clips labeled
# good / degraded / insufficient once we have one — see the "Limitations"
# note in the README. Treat every number below as provisional.
#
#   speech_ratio < 0.15                       -> "insufficient"
#   OR speech_duration_s < 0.5                -> "insufficient"
#   OR total_duration_s  < 0.5                -> "insufficient"
#   0.15 <= speech_ratio < 0.5                -> "degraded"
#   speech_ratio >= 0.5                       -> "good"
#
# Reasoning:
#   * speech_ratio >= 0.5 ("good"): at least half the clip is voiced. That
#     is what a person speaking into the phone with only normal gaps looks
#     like — mostly continuous speech, enough acoustic material for the
#     model to pool a stable embedding over.
#   * 0.15 <= speech_ratio < 0.5 ("degraded"): there is real speech, but
#     it is a minority of the clip. Typically means a noisy/echoey line,
#     someone talking intermittently, long hold gaps, or heavy cross-talk
#     that the GMM VAD only partially catches. The model can still say
#     something, but the caller should weight it less — hence a distinct
#     bucket rather than lumping it in with "good".
#   * speech_ratio < 0.15 ("insufficient"): essentially silence, hold
#     music, line noise, or a mis-triggered capture. Not enough voiced
#     speech to say anything meaningful; running the model here mostly
#     produces confident-looking noise, so we skip it.
#   * The two absolute floors (speech_duration_s < 0.5, total_duration_s
#     < 0.5) catch the case where the *ratio* looks fine but there is
#     barely any audio at all — e.g. a 0.3 s clip that is 100% speech.
#     Half a second is roughly the shortest span from which this model
#     gives a non-random age/gender read. It also mirrors the sub-frame
#     guard in run_vad(), which returns speech_ratio 0.0 for clips shorter
#     than one 30 ms VAD frame.
_MIN_SPEECH_RATIO_USABLE = 0.15
_MIN_SPEECH_RATIO_GOOD = 0.5
_MIN_SPEECH_DURATION_S = 0.5
_MIN_TOTAL_DURATION_S = 0.5


def assess_quality(vad_stats: dict) -> str:
    """Map VAD stats onto a ``good`` / ``degraded`` / ``insufficient`` label.

    Args:
        vad_stats: The dict returned by :func:`app.audio.ingest.run_vad`.
            Uses ``speech_ratio``, ``speech_duration_s`` and
            ``total_duration_s``; ``num_speech_segments`` is accepted but
            not currently part of the decision. Missing keys are treated
            as ``0.0`` so a partial/empty dict degrades to
            ``"insufficient"`` rather than raising.

    Returns:
        One of the string values of
        :class:`app.schemas.models.AudioQuality`: ``"good"``,
        ``"degraded"``, or ``"insufficient"``. A return of
        ``"insufficient"`` is the signal to the caller to skip inference
        entirely and respond with ``unknown`` predictions.
    """
    speech_ratio = float(vad_stats.get("speech_ratio", 0.0) or 0.0)
    speech_duration_s = float(vad_stats.get("speech_duration_s", 0.0) or 0.0)
    total_duration_s = float(vad_stats.get("total_duration_s", 0.0) or 0.0)

    # Not enough real speech to say anything meaningful.
    if (
        speech_ratio < _MIN_SPEECH_RATIO_USABLE
        or speech_duration_s < _MIN_SPEECH_DURATION_S
        or total_duration_s < _MIN_TOTAL_DURATION_S
    ):
        return AudioQuality.INSUFFICIENT.value

    # Some speech, but a minority of the clip: likely noisy/intermittent.
    if speech_ratio < _MIN_SPEECH_RATIO_GOOD:
        return AudioQuality.DEGRADED.value

    # Clear, mostly continuous speech.
    return AudioQuality.GOOD.value
