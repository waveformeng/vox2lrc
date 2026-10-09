from collections.abc import Callable
from dataclasses import dataclass

from . import audio
from .align import Alignment, align_lyrics, keywords_from
from .engines import Engine
from .lines import Line, group_lines
from .types import Word

# Below this share of lyric words heard exactly, the lyrics probably aren't
# this song: the plain transcription is returned instead (same rule as the app).
MIN_MATCH_RATIO = 0.3


@dataclass
class Transcript:
    words: list[Word]
    duration: float
    language: str | None
    engine: str
    chunks: list[audio.Span]
    alignment: Alignment | None = None  # set when lyrics were given

    def lines(self, max_gap: float = 0.6, max_words: int = 10) -> list[Line]:
        """The caller's lyric lines when they aligned, else lines grouped from the transcription."""
        if self.alignment and self.alignment.ratio >= MIN_MATCH_RATIO:
            return self.alignment.lines
        return group_lines(self.words, max_gap=max_gap, max_words=max_words)

    def alignment_report(self) -> dict | None:
        a = self.alignment
        if a is None:
            return None
        return {"used": a.ratio >= MIN_MATCH_RATIO, "matched": a.matched, "total": a.total, "ratio": round(a.ratio, 3)}


def transcribe_stem(path: str, engine: Engine, language: str | None = None, *,
                    lyrics: str | None = None,
                    max_duration: float = 15 * 60, noise_db: float = -35.0,
                    on_chunk: Callable[[audio.Span, list[Word]], None] | None = None) -> Transcript:
    """Transcribe a vocal stem chunk by chunk and return words in stem time.

    With lyrics, their words are passed to the engine as hints and the
    transcription is aligned to them: the lyrics give the words and lines, the
    audio gives the timing (see align.py).

    max_duration rejects over-long uploads before any decoding work is done.
    """
    duration = audio.probe_duration(path)
    if duration > max_duration:
        raise ValueError(f"audio is {duration:.0f} s, over the {max_duration:.0f} s limit")
    silences = audio.detect_silences(path, duration, noise_db=noise_db)
    chunks = audio.plan_chunks(duration, silences)

    hints = keywords_from(lyrics) if lyrics else None
    words: list[Word] = []
    for chunk, pcm in audio.iter_chunk_pcm(path, chunks):
        chunk_words = [_clamp(w.shifted(chunk.start), chunk) for w in engine.transcribe(pcm, language, hints)]
        if on_chunk:
            on_chunk(chunk, chunk_words)
        words.extend(chunk_words)
    words.sort(key=lambda w: w.start)
    alignment = align_lyrics(lyrics, words, duration) if lyrics else None
    return Transcript(words, duration, language, engine.name, chunks, alignment)


def _clamp(w: Word, span: audio.Span) -> Word:
    start = min(max(w.start, span.start), span.end)
    end = min(max(w.end, start), span.end)
    return Word(w.word, start, end, w.probability)
