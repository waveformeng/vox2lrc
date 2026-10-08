from dataclasses import dataclass

from .types import Word

_SENTENCE_END = (".", "!", "?")


@dataclass(frozen=True)
class Line:
    words: tuple[Word, ...]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def group_lines(words: list[Word], max_gap: float = 0.6, max_words: int = 10, min_words: int = 3) -> list[Line]:
    """Split a word stream into lyric lines.

    A new line starts when the pause before a word exceeds max_gap, when the
    current line already has max_words, or after sentence-ending punctuation
    once the line has at least min_words.
    """
    lines: list[Line] = []
    current: list[Word] = []
    for w in words:
        if current:
            prev = current[-1]
            if (
                w.start - prev.end > max_gap
                or len(current) >= max_words
                or (prev.word.endswith(_SENTENCE_END) and len(current) >= min_words)
            ):
                lines.append(Line(tuple(current)))
                current = []
        current.append(w)
    if current:
        lines.append(Line(tuple(current)))
    return lines
