"""Speech-to-text backends. Each takes one chunk of 16 kHz mono float32 PCM.

Word times returned by a backend are relative to the start of the chunk.
"""

from typing import Protocol

from .types import Word


class Engine(Protocol):
    name: str

    def transcribe(self, pcm: bytes, language: str | None, hints: list[str] | None = None) -> list[Word]:
        """hints: words expected in the audio (from known lyrics). Engines may ignore them."""
        ...


class WhistleEngine:
    """Cactus Whistle (cactus-needle): 16.9 MB, CPU-only, at most 30 s per call.

    Uses the Whistle class directly; the module-level needle.transcribe() also
    sends an anonymous usage ping. Weights and the engine library are fetched
    from Hugging Face into ~/.cache/cactus-needle on first use.
    """

    name = "whistle"

    def __init__(self, weights: str | None = None):
        import needle

        self._model = needle.Whistle(weights=weights)

    def transcribe(self, pcm: bytes, language: str | None, hints: list[str] | None = None) -> list[Word]:
        # Hints become Whistle's keyword biasing, which measurably improves how
        # many lyric words it hears.
        result = self._model.transcribe(pcm, language=language, keywords=hints or None, word_timestamps=True)
        return [
            Word(w["word"].strip(), float(w["start"]), float(w["end"]), float(w["probability"]))
            for w in result.get("words", [])
            if w["word"].strip()
        ]


class FasterWhisperEngine:
    """faster-whisper on CPU (int8). small/medium need roughly 2-4 GB RAM."""

    def __init__(self, model: str = "small", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper-{model}"
        self._model = WhisperModel(model, device="cpu", compute_type=compute_type)

    def transcribe(self, pcm: bytes, language: str | None, hints: list[str] | None = None) -> list[Word]:
        # Hints are ignored: prompting Whisper with lyrics makes it write them out
        # whether or not they're sung, which measured worse timing.
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.float32)
        segments, _ = self._model.transcribe(
            samples, language=language, word_timestamps=True, vad_filter=False,
            condition_on_previous_text=False,
        )
        return [
            Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
            for seg in segments
            for w in (seg.words or [])
            if w.word.strip()
        ]


def load_engine(spec: str) -> Engine:
    """'whistle', or 'faster-whisper' / 'faster-whisper:<model>' (e.g. faster-whisper:medium)."""
    name, _, arg = spec.partition(":")
    if name == "whistle":
        return WhistleEngine()
    if name == "faster-whisper":
        return FasterWhisperEngine(arg or "small")
    raise ValueError(f"unknown engine {spec!r}: use 'whistle' or 'faster-whisper[:model]'")
