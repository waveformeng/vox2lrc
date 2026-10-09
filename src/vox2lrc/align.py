"""Line up transcribed words with lyrics the caller already has.

The lyrics are the words; the transcription is the timing. Each lyric word is
paired with a transcribed word (same, similar or replaced) or with nothing, by
a word-level edit-distance alignment. Paired words take the transcribed
word's times; unpaired ones share out the gap between their timed neighbours
by length. Transcribed words with no lyric counterpart are dropped. The
output keeps the caller's lines exactly.

Same algorithm and costs as waveform-karaoke-open's src/lib/lyrics-align.ts.
"""

import re
import unicodedata
from dataclasses import dataclass

from .lines import Line
from .types import Word

MAX_LYRICS_LENGTH = 20_000
# Seconds per word when words before the first or after the last timed word need placing.
FALLBACK_WORD_SECONDS = 0.35

_GAP = 1.0
_NON_WORD = re.compile(r"[^\w]|_", re.UNICODE)


def parse_lyrics(text: str) -> list[str]:
    """Lyrics text → cleaned lines, without blank lines."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return [line for line in lines if line]


def keywords_from(lyrics: str, limit: int = 200) -> list[str]:
    """Distinct lyric words for an engine's keyword biasing, longest first
    (short words are common anyway and need no help)."""
    seen: dict[str, str] = {}
    for line in parse_lyrics(lyrics):
        for word in line.split(" "):
            clean = word.strip(".,!?;:\"'()[]")
            if len(clean) > 3 and clean.lower() not in seen:
                seen[clean.lower()] = clean
    return sorted(seen.values(), key=len, reverse=True)[:limit]


def normalize_word(word: str) -> str:
    """How words are compared: case, accents, punctuation and apostrophes ignored."""
    decomposed = unicodedata.normalize("NFKD", word)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_WORD.sub("", stripped.lower())


def similarity(a: str, b: str) -> float:
    """1 - edit distance / longer length: 1 for equal, 0 for nothing in common."""
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


def _substitution(a: str, b: str) -> tuple[float, str]:
    if a == b:
        return 0.0, "same"
    if a and b and similarity(a, b) >= 0.5:
        return 0.5, "similar"
    return 1.5, "replaced"


def pair_words(ref: list[str], hyp: list[str]) -> list[tuple[int, str]]:
    """For each lyric word: (index of its transcribed word or -1, kind)."""
    n, m = len(ref), len(hyp)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]  # 0 pair, 1 lyric word alone, 2 extra transcribed word
    for i in range(1, n + 1):
        cost[i][0], move[i][0] = i * _GAP, 1
    for j in range(1, m + 1):
        cost[0][j], move[0][j] = j * _GAP, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, d = cost[i - 1][j - 1] + _substitution(ref[i - 1], hyp[j - 1])[0], 0
            if (up := cost[i - 1][j] + _GAP) < best:
                best, d = up, 1
            if (left := cost[i][j - 1] + _GAP) < best:
                best, d = left, 2
            cost[i][j], move[i][j] = best, d

    out: list[tuple[int, str]] = [(-1, "missing")] * n
    i, j = n, m
    while i > 0:
        d = 1 if j == 0 else move[i][j]
        if d == 0:
            out[i - 1] = (j - 1, _substitution(ref[i - 1], hyp[j - 1])[1])
            i, j = i - 1, j - 1
        elif d == 1:
            i -= 1
        else:
            j -= 1
    return out


@dataclass
class Alignment:
    lines: list[Line]
    matched: int  # lyric words the transcription got exactly
    total: int  # all lyric words
    gaps: list[tuple[int, int]]  # runs of lyric words [start, end) with no transcribed word

    @property
    def ratio(self) -> float:
        return self.matched / self.total if self.total else 0.0


def align_lyrics(lyrics: str, words: list[Word], duration: float | None) -> Alignment | None:
    """The caller's lyrics, timed from the transcribed words. None if either is empty."""
    lines = [line.split(" ") for line in parse_lyrics(lyrics)]
    ref = [w for line in lines for w in line]
    if not ref or not words:
        return None

    pairs = pair_words([normalize_word(w) for w in ref], [normalize_word(w.word) for w in words])
    slots: list[tuple[float, float, float] | None] = []
    for j, kind in pairs:
        if j < 0:
            slots.append(None)
        else:
            h = words[j]
            slots.append((h.start, h.end, h.probability if kind == "same" else min(h.probability, 0.5)))
    gaps = _fill_gaps(slots, ref, duration)

    out: list[Line] = []
    k = 0
    for line in lines:
        timed = []
        for text in line:
            start, end, p = slots[k]  # type: ignore[misc]
            timed.append(Word(text, round(start, 3), round(end, 3), p))
            k += 1
        out.append(Line(tuple(timed)))
    matched = sum(1 for _, kind in pairs if kind == "same")
    return Alignment(out, matched, len(ref), gaps)


def _fill_gaps(slots: list, words: list[str], duration: float | None) -> list[tuple[int, int]]:
    gaps = []
    i = 0
    while i < len(slots):
        if slots[i] is not None:
            i += 1
            continue
        j = i
        while j < len(slots) and slots[j] is None:
            j += 1
        gaps.append((i, j))
        count = j - i
        prev = slots[i - 1][1] if i > 0 else None
        nxt = slots[j][0] if j < len(slots) else None
        lo = prev if prev is not None else max(0.0, nxt - count * FALLBACK_WORD_SECONDS)
        hi = nxt if nxt is not None else lo + count * FALLBACK_WORD_SECONDS
        if nxt is None and duration is not None and duration > lo:
            hi = min(hi, duration)
        if hi < lo:
            lo = hi
        lengths = [max(1, len(normalize_word(w))) for w in words[i:j]]
        total = sum(lengths)
        at = lo
        for k in range(i, j):
            start = at
            at += (hi - lo) * lengths[k - i] / total
            slots[k] = (start, at, 0.0)
        i = j
    return gaps
