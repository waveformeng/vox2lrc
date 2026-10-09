"""Same cases as waveform-karaoke-open's src/lib/lyrics-align.test.ts, so the two stay in step."""

from vox2lrc.align import align_lyrics, keywords_from, normalize_word, parse_lyrics
from vox2lrc.types import Word


def transcribed(*words: str, at: float = 10.0) -> list[Word]:
    """Each word 0.5 s long, back to back from `at`."""
    return [Word(w, at + i * 0.5, at + (i + 1) * 0.5, 0.9) for i, w in enumerate(words)]


def flat(a):
    return [w for line in a.lines for w in line.words]


def test_parse_lyrics():
    assert parse_lyrics("  Take me out\r\n\n\nto the   ball game,\n") == ["Take me out", "to the ball game,"]


def test_normalize_word():
    assert normalize_word("Don't,") == "dont"
    assert normalize_word("Café!") == "cafe"
    assert normalize_word("—") == ""


def test_uses_lyric_words_and_lines_with_transcribed_timing():
    words = transcribed("take", "me", "out", "to", "the", "ball", "game")
    a = align_lyrics("Take me out\nto the ball game,", words, 60)
    assert [line.text for line in a.lines] == ["Take me out", "to the ball game,"]
    assert [(w.word, w.start, w.end) for w in flat(a)] == [
        (t, w.start, w.end) for t, w in zip(["Take", "me", "out", "to", "the", "ball", "game,"], words)
    ]
    assert a.lines[1].start == 11.5
    assert a.ratio == 1


def test_fixes_misheard_words_in_place():
    a = align_lyrics("take me out to the ball game", transcribed("take", "me", "out", "to", "the", "fall", "gain"), 60)
    fixed = flat(a)
    assert [w.word for w in fixed] == ["take", "me", "out", "to", "the", "ball", "game"]
    assert (fixed[5].start, fixed[5].end) == (12.5, 13)
    assert fixed[5].probability <= 0.5 and fixed[0].probability == 0.9


def test_times_missed_words_between_neighbours():
    words = transcribed("take", "me", "out", "to") + [w.shifted(1) for w in transcribed("ball", "game", at=12)]
    the = flat(align_lyrics("take me out to the ball game", words, 60))[4]
    assert (the.word, the.start, the.end, the.probability) == ("the", 12, 13, 0)


def test_drops_imagined_words():
    a = align_lyrics("take me out", transcribed("take", "me", "uh", "out", "yeah"), 60)
    assert [w.word for w in flat(a)] == ["take", "me", "out"]
    assert flat(a)[2].start == 11.5


def test_places_words_before_first_and_after_last():
    intro, middle, outro = flat(align_lyrics("intro middle outro", transcribed("middle"), 60))
    assert intro.end == middle.start and intro.start >= 0
    assert outro.start == middle.end and outro.end <= 60


def test_times_stay_in_order():
    a = align_lyrics("a x b\nc\ny z\nf", transcribed("a", "b", "c", "d", "e", "f"), 60)
    starts = [w.start for w in flat(a)]
    assert starts == sorted(starts)
    assert all(w.end >= w.start for w in flat(a))


def test_reports_match_ratio_and_gaps():
    a = align_lyrics("completely different words about the sea",
                     transcribed("take", "me", "out", "to", "the", "ball", "game"), 60)
    assert a.ratio < 0.3


def test_nothing_to_align():
    assert align_lyrics("\n  \n", transcribed("a"), 60) is None
    assert align_lyrics("a", [], 60) is None


def test_keywords_from_lyrics():
    kws = keywords_from("Buy me some peanuts and Cracker Jacks,\nI don't care if I never get back\nPeanuts!")
    assert kws[:2] == ["peanuts", "Cracker"]  # longest first, ties in lyric order
    assert "Jacks" in kws and "peanuts" in kws and "Peanuts" not in kws  # distinct, case-insensitive
    assert "me" not in kws and "and" not in kws  # short words need no help
