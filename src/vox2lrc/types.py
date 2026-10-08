from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """One transcribed word. Times are seconds from the start of the stem."""

    word: str
    start: float
    end: float
    probability: float

    def shifted(self, offset: float) -> "Word":
        return Word(self.word, self.start + offset, self.end + offset, self.probability)
