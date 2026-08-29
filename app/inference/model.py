"""Voice attribute inference: wav2vec2 age/gender model wrapper.

This module owns the acoustic model that turns a decoded waveform into a
speaker gender guess and a coarse age bracket. It is deliberately thin: it
loads one checkpoint once, and every ``predict()`` call is a pure function
of its arguments plus the frozen weights (no per-request state is cached).

Why this model
--------------
We use ``audeering/wav2vec2-large-robust-24-ft-age-gender``. Two reasons,
both of which matter for a call-center / telephony workload:

* **Robust pre-training.** The backbone is wav2vec2-large-*robust*, which
  was pre-trained on a deliberately messy mix (Libri-Light read speech,
  CommonVoice, Switchboard and Fisher *telephone* speech, plus noisy
  in-the-wild audio). It degrades gracefully on 8 kHz G.711 telephone
  audio, codec artefacts, background noise and cross-talk -- exactly the
  conditions our ingestion layer hands it after upsampling SIP-trunk
  audio to 16 kHz. A model fine-tuned only on clean studio speech tends
  to collapse toward a single prediction on that input.
* **Single forward pass, two heads.** One backbone emits a pooled
  embedding; a regression head predicts age (0-1, mapping to 0-100 years)
  and a 3-class head predicts gender (child / female / male). We get both
  attributes for the cost of one inference, which keeps p95 latency low.

The checkpoint ships a *custom* architecture that is not part of
``transformers``; :class:`_AgeGenderModel` / :class:`_ModelHead` below are
the model-card definitions, needed so ``from_pretrained`` has something to
load the weights into.

Age-bucket mapping
------------------
The age head is a **regression** in [0, 1] that the model card maps to
0-100 years (``years = value * 100``). The API contract, however, wants
one of four coarse brackets (``18-30`` / ``31-45`` / ``46-60`` / ``60+``)
plus ``unknown``. :meth:`AttributeInferencer.predict` does that bucketing.
See :data:`_AGE_BRACKET_BOUNDS` for the exact cut points and the reasoning
behind them.

Requires ``torch``, ``transformers`` and ``numpy``. The model weights
(~1.3 GB) download once into ``.model_cache/`` and are memory-resident for
the process lifetime.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
)

MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"
TARGET_SAMPLE_RATE = 16_000

# Weights are cached here so the first process to run pays the download and
# every process after that loads from local disk. Kept inside the repo (and
# .gitignored) so it survives across container rebuilds when mounted.
CACHE_DIR = Path(__file__).resolve().parents[2] / ".model_cache"

# --- gender head label order -------------------------------------------------
# Per the model card, the 3-class gender softmax is ordered:
#   index 0 -> child, 1 -> female, 2 -> male
# The API only has male / female / unknown, so "child" (and anything we are
# not sure how to name) folds into "unknown" -- see _map_gender().
_GENDER_LABELS = ("child", "female", "male")

# --- age regression -> bracket mapping ------------------------------------
#
# The regression head outputs v in [0, 1]; the model card's convention is
#   age_years = v * 100
# so the raw value is effectively "fraction of a 100-year lifespan".
#
# We collapse that continuous age onto the four contract brackets using the
# brackets' own upper edges as cut points -- i.e. bucket by the predicted
# year, nothing cleverer:
#
#   predicted years  ->  bracket
#   ---------------------------------
#   [18, 30]         ->  "18-30"     (v in [0.18, 0.30])
#   (30, 45]         ->  "31-45"     (v in (0.30, 0.45])
#   (45, 60]         ->  "46-60"     (v in (0.45, 0.60])
#   > 60             ->  "60+"       (v > 0.60)
#   < 18             ->  "unknown"   (see below)
#
# Reasoning for the cut points:
#   * The brackets in app/schemas/models.py are contiguous with no gap
#     (18-30, 31-45, 46-60, 60+), so a single-year boundary decision is
#     unambiguous once we pick which side of 30/45/60 is inclusive. We put
#     the boundary year in the *lower* bracket (<= 30 is "18-30") because
#     that matches how people say ranges ("early thirties" starts at 31).
#   * We do NOT try to widen bands near the boundaries or add hysteresis.
#     The downstream consumer is coarse call-center analytics; a speaker
#     whose true age sits on a boundary being assigned to either
#     neighbouring bracket is acceptable, and a deterministic rule is
#     easier to reason about and test.
#   * Predicted age < 18 -> "unknown" rather than forcing "18-30". The
#     product only cares about adult callers; a sub-18 regression output
#     usually means the voice is childlike, out-of-distribution, or the
#     clip is too degraded for the head to commit -- none of which we want
#     to report as a confident "18-30".
#
# Confidence for the age bracket: the regression head emits a single
# scalar with NO posterior, so there is no honest probability to report.
# We surface a deliberately modest proxy instead: how far the predicted
# year sits from the nearest bracket edge, as a fraction of half the
# bracket width, then squeezed into the band [_AGE_CONF_FLOOR,
# _AGE_CONF_CEIL]. A prediction on a bracket edge scores the floor
# (~0.30); one dead-centre scores the ceiling (~0.65). It never reaches
# 1.0 because we genuinely do not know the model is that sure -- e.g.
# silence regresses to a mid-bracket year but tells us nothing. Callers
# should read this as "roughly how interior to the bucket is the point
# estimate", not as a calibrated confidence. Documented again on
# predict().
_AGE_MIN_YEARS = 18.0
_AGE_CONF_FLOOR = 0.30
_AGE_CONF_CEIL = 0.65
_AGE_BRACKET_BOUNDS = (
    # (label, lower_year_inclusive, upper_year_inclusive_or_None_for_open)
    ("18-30", 18.0, 30.0),
    ("31-45", 30.0, 45.0),
    ("46-60", 45.0, 60.0),
    ("60+", 60.0, None),
)


class _ModelHead(nn.Module):
    """Dense -> tanh -> dense head on top of the pooled hidden state.

    Copied from the ``audeering`` model card; used for both the 1-unit age
    regression head and the 3-unit gender classification head.
    """

    def __init__(self, config, num_labels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):  # noqa: D102 - see class docstring
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGenderModel(Wav2Vec2PreTrainedModel):
    """wav2vec2-robust backbone with an age head and a gender head.

    ``forward(input_values)`` returns a 3-tuple:
      * ``hidden_states`` -- pooled embedding, shape ``[batch, hidden_size]``
      * ``logits_age``    -- regression scalar in ~[0, 1], shape ``[batch, 1]``
      * ``logits_gender`` -- softmax probs over (child, female, male),
        shape ``[batch, 3]``

    Definition copied verbatim from the model card so ``from_pretrained``
    can populate it.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _ModelHead(config, 1)
        self.gender = _ModelHead(config, 3)
        # The model card calls the legacy ``self.init_weights()`` here. On
        # transformers >= 5 that no longer wires up bookkeeping that
        # ``from_pretrained`` later relies on (``all_tied_weights_keys``),
        # so use ``post_init()``, which does weight init *and* that setup.
        self.post_init()

    def forward(self, input_values):  # noqa: D102 - see class docstring
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


class AttributeInferencer:
    """Loads the age/gender model once and serves per-clip predictions.

    The model weights are the *only* state held across calls. Construct this
    once on service startup (it downloads / loads ~1.3 GB) and reuse the
    instance for every request; :meth:`predict` keeps nothing from one call
    to the next.

    Example::

        inferencer = AttributeInferencer()          # once, on startup
        result = inferencer.predict(waveform, 16000)  # per request
        # -> {"gender_prediction": "female", "gender_confidence": 0.97,
        #     "age_bracket": "31-45", "age_confidence": 0.72,
        #     "inference_ms": 84.3}
    """

    def __init__(self, cache_dir: Path | str = CACHE_DIR) -> None:
        """Load the processor and model into memory.

        Args:
            cache_dir: Directory to download / read the HF weights from.
                Defaults to ``<repo>/.model_cache``. The directory is
                created if missing.

        Raises:
            RuntimeError: If the processor or model cannot be loaded (no
                network on a cold cache, corrupt cache, out of disk/RAM).
                Raised here, at startup, on purpose -- a service that
                cannot load its model should fail fast, not fail per
                request.
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._processor = Wav2Vec2Processor.from_pretrained(
                MODEL_NAME, cache_dir=str(self._cache_dir)
            )
            self._model = _AgeGenderModel.from_pretrained(
                MODEL_NAME, cache_dir=str(self._cache_dir)
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as RuntimeError
            raise RuntimeError(
                f"failed to load {MODEL_NAME} from {self._cache_dir}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._model.eval()
        # CPU-only: the workload is short clips and this keeps deployment
        # simple. Move to CUDA here if a GPU node is ever used.
        self._device = torch.device("cpu")
        self._model.to(self._device)

    def predict(
        self, waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE
    ) -> dict:
        """Infer speaker gender and age bracket for one clip.

        The whole model forward pass is wrapped in try/except: on *any*
        failure (bad input shape, NaNs, an OOM, a transformers internal
        error) this returns ``unknown`` predictions with ``0.0``
        confidence rather than propagating the exception. The service
        should degrade, not 500, on a single unusable clip.

        Args:
            waveform: Mono ``float32`` waveform in [-1, 1], as produced by
                ``app.audio.ingest.normalize_audio``. A 2-D ``[1, N]``
                array is accepted and squeezed.
            sample_rate: Sample rate of ``waveform`` in Hz. The model
                expects 16 kHz; other rates are passed to the feature
                extractor, which will resample-or-warn, but callers should
                normalize upstream.

        Returns:
            A dict with:

            * ``gender_prediction`` (str): ``"male"`` | ``"female"`` |
              ``"unknown"``. ``"unknown"`` when the top gender class is
              "child" or when the forward pass failed.
            * ``gender_confidence`` (float): the softmax probability of the
              reported class, in [0, 1]. ``0.0`` on failure or when the
              prediction is ``"unknown"`` via the child class (we do not
              surface the child probability as a male/female confidence).
            * ``age_bracket`` (str): ``"18-30"`` | ``"31-45"`` |
              ``"46-60"`` | ``"60+"`` | ``"unknown"``. ``"unknown"`` when
              the predicted age is < 18 years or the forward pass failed.
            * ``age_confidence`` (float): a deliberately modest *heuristic*
              in roughly [0.30, 0.65] for how interior to its bracket the
              predicted age sits -- ~0.30 on a bracket edge, ~0.65 dead
              centre. It is NOT a calibrated probability and never nears
              1.0: the regression head emits no posterior, and a
              mid-bracket value (all that silence or noise produces)
              must not read as high confidence. ``0.0`` when the bracket
              is ``"unknown"``.
            * ``inference_ms`` (float): wall-clock milliseconds spent in
              the model forward pass *only*. Feature extraction / tensor
              setup happen before the timer starts and bracket mapping
              after it stops, so this number is comparable across clips of
              different lengths' preprocessing cost.
        """
        failed = {
            "gender_prediction": "unknown",
            "gender_confidence": 0.0,
            "age_bracket": "unknown",
            "age_confidence": 0.0,
            "inference_ms": 0.0,
        }

        try:
            signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if signal.size == 0:
                # Nothing to run the model on. Treat like a failed pass:
                # unknown / 0.0, and don't bother the model.
                return dict(failed)

            # --- preprocessing (deliberately outside the timed region) ---
            proc = self._processor(signal, sampling_rate=sample_rate)
            input_values = torch.from_numpy(
                np.asarray(proc["input_values"][0], dtype=np.float32).reshape(1, -1)
            ).to(self._device)

            # --- timed forward pass -------------------------------------
            start = time.perf_counter()
            with torch.no_grad():
                _hidden, logits_age, logits_gender = self._model(input_values)
            inference_ms = (time.perf_counter() - start) * 1000.0

            # --- post-processing (outside the timed region) ------------
            age_value = float(logits_age.squeeze().item())
            gender_probs = logits_gender.squeeze().tolist()

            gender_prediction, gender_confidence = _map_gender(gender_probs)
            age_bracket, age_confidence = _map_age(age_value)

            return {
                "gender_prediction": gender_prediction,
                "gender_confidence": gender_confidence,
                "age_bracket": age_bracket,
                "age_confidence": age_confidence,
                "inference_ms": inference_ms,
            }
        except Exception:  # noqa: BLE001 - contract: degrade, never crash
            return dict(failed)


def _map_gender(probs: list[float]) -> tuple[str, float]:
    """Map a (child, female, male) softmax to (label, confidence).

    * ``female`` / ``male`` top class -> that label, with its probability
      as the confidence.
    * ``child`` top class -> ``"unknown"`` with ``0.0`` confidence. The
      API has no "child" value and the child probability is not a
      male/female confidence, so we do not surface it. A childlike or
      out-of-distribution voice being reported as ``unknown`` is the
      intended, conservative behaviour.
    * Malformed input (wrong length, non-finite) -> ``"unknown"``, ``0.0``.
    """
    if len(probs) != len(_GENDER_LABELS) or not all(np.isfinite(probs)):
        return "unknown", 0.0

    top_idx = int(np.argmax(probs))
    label = _GENDER_LABELS[top_idx]
    if label in ("male", "female"):
        return label, float(probs[top_idx])
    # label == "child" (or any future 4th class we don't recognise)
    return "unknown", 0.0


def _map_age(value: float) -> tuple[str, float]:
    """Map the raw age regression value in [0, 1] to (bracket, confidence).

    ``value`` is interpreted as ``age_years = value * 100`` (model-card
    convention). See the module-level comment on ``_AGE_BRACKET_BOUNDS``
    for the bracket cut points and rationale.

    Confidence is a deliberately modest proxy, NOT a calibrated
    probability -- the regression head has no posterior. It is the
    predicted year's distance from the nearest edge of its bracket, as a
    fraction of half the bracket width (0 on an edge, 1 dead-centre),
    linearly mapped into ``[_AGE_CONF_FLOOR, _AGE_CONF_CEIL]``. It never
    reaches 1.0: a mid-bracket regression output (which is all silence or
    noise produces) should not read as high confidence. For the
    open-ended ``60+`` bracket we use a nominal 30-year span (60-90) for
    the width.
    """
    if not np.isfinite(value):
        return "unknown", 0.0

    years = value * 100.0

    if years < _AGE_MIN_YEARS:
        # Below the youngest bracket -> not an adult caller we score.
        return "unknown", 0.0

    for label, lo, hi in _AGE_BRACKET_BOUNDS:
        upper = hi if hi is not None else lo + 30.0  # nominal span for "60+"
        # Lower bound is exclusive except for the very first bracket, so a
        # boundary year lands in the lower bracket (<= 30 -> "18-30").
        in_bracket = (years >= lo if label == "18-30" else years > lo) and (
            years <= upper or hi is None
        )
        if in_bracket:
            centre = (lo + upper) / 2.0
            half_width = (upper - lo) / 2.0
            # interiority: 0 on the nearest edge, 1 at the centre.
            interiority = 1.0 - abs(years - centre) / half_width
            interiority = min(1.0, max(0.0, interiority))
            # squeeze into the modest confidence band.
            confidence = _AGE_CONF_FLOOR + interiority * (
                _AGE_CONF_CEIL - _AGE_CONF_FLOOR
            )
            return label, float(confidence)

    # years >= _AGE_MIN_YEARS but fell through every bracket: only possible
    # if the bounds table is edited inconsistently. Be safe.
    return "unknown", 0.0
