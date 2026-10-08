from collections.abc import Callable
from dataclasses import dataclass

from . import audio
from .engines import Engine
from .types import Word


@dataclass
class Transcript:
    words: list[Word]
    duration: float
    language: str | None
    engine: str
    chunks: list[audio.Span]


def transcribe_stem(path: str, engine: Engine, language: str | None = None, *,
                    max_duration: float = 15 * 60, noise_db: float = -35.0,
                    on_chunk: Callable[[audio.Span, list[Word]], None] | None = None) -> Transcript:
    """Transcribe a vocal stem chunk by chunk and return words in stem time.

    max_duration rejects over-long uploads before any decoding work is done.
    """
    duration = audio.probe_duration(path)
    if duration > max_duration:
        raise ValueError(f"audio is {duration:.0f} s, over the {max_duration:.0f} s limit")
    silences = audio.detect_silences(path, duration, noise_db=noise_db)
    chunks = audio.plan_chunks(duration, silences)

    words: list[Word] = []
    for chunk, pcm in audio.iter_chunk_pcm(path, chunks):
        chunk_words = [_clamp(w.shifted(chunk.start), chunk) for w in engine.transcribe(pcm, language)]
        if on_chunk:
            on_chunk(chunk, chunk_words)
        words.extend(chunk_words)
    words.sort(key=lambda w: w.start)
    return Transcript(words, duration, language, engine.name, chunks)


def _clamp(w: Word, span: audio.Span) -> Word:
    start = min(max(w.start, span.start), span.end)
    end = min(max(w.end, start), span.end)
    return Word(w.word, start, end, w.probability)
