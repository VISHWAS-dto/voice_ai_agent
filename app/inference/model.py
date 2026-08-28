"""Voice attribute inference model wrapper.

Stub only. This module loads the acoustic model(s) and produces gender and
age-bracket predictions from a decoded waveform.

Planned approach: a ``transformers`` audio classification backbone (e.g. a
wav2vec2 / HuBERT variant) fine-tuned for speaker traits, run via ``torch``
on CPU or GPU. A single model with two heads, or two models, TBD.
"""

from dataclasses import dataclass

from app.audio.ingest import DecodedAudio
from app.schemas.models import AgeBracketResult, GenderResult


@dataclass
class InferenceResult:
    """Bundled model output for one clip."""

    gender: GenderResult
    age_bracket: AgeBracketResult


class VoiceAttributeModel:
    """Loads weights once and serves gender + age-bracket predictions."""

    def __init__(self, model_dir: str, *, device: str = "cpu") -> None:
        """Prepare the model wrapper.

        Args:
            model_dir: Path or HF hub id for the fine-tuned checkpoint(s).
            device: Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.

        Note:
            Weights are loaded lazily in :meth:`load` so construction stays
            cheap at import time.
        """
        raise NotImplementedError

    def load(self) -> None:
        """Load model weights and feature extractor into memory.

        Idempotent: safe to call multiple times; subsequent calls are no-ops.
        """
        raise NotImplementedError

    def predict(self, audio: DecodedAudio) -> InferenceResult:
        """Run inference on a single decoded clip.

        Args:
            audio: Canonical decoded audio that has passed quality gating.

        Returns:
            An :class:`InferenceResult` with per-attribute predictions and
            confidences.

        Raises:
            RuntimeError: If called before :meth:`load`.
        """
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        """Whether :meth:`load` has completed and the model is ready."""
        raise NotImplementedError
