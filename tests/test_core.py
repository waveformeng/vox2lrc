from vox2lrc.audio import Span, parse_silencedetect, plan_chunks
from vox2lrc.lines import group_lines
from vox2lrc.lrc import format_timestamp, to_json, to_lrc
from vox2lrc.types import Word


def w(text, start, end, p=0.9):
    return Word(text, start, end, p)


def test_parse_silencedetect_with_trailing_silence():
    stderr = """
[silencedetect @ 0x1] silence_start: -0.002
[silencedetect @ 0x1] silence_end: 4.5 | silence_duration: 4.5
[silencedetect @ 0x1] silence_start: 60.25
"""
    assert parse_silencedetect(stderr, duration=62.0) == [Span(0.0, 4.5), Span(60.25, 62.0)]


def test_plan_chunks_merges_voiced_regions_and_drops_silence():
    silences = [Span(0, 4), Span(10, 11), Span(20, 25), Span(40, 50)]
    chunks = plan_chunks(50, silences, pad=0.0)
    # 4-10 + 11-20 fit in one chunk (16 s); adding 25-40 would make it 36 s.
    assert chunks == [Span(4, 20), Span(25, 40)]


def test_plan_chunks_splits_long_voiced_region():
    chunks = plan_chunks(70, [], max_len=29.5, pad=0.0)
    assert len(chunks) == 3
    assert all(c.duration <= 29.5 for c in chunks)
    assert chunks[0].start == 0 and chunks[-1].end == 70


def test_plan_chunks_padding_stays_within_limits():
    silences = [Span(5.0, 5.2)]
    chunks = plan_chunks(40, silences, max_len=29.5, pad=0.15)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start
    assert all(c.duration <= 29.5 for c in chunks)


def test_group_lines_on_gap_punctuation_and_length():
    words = [
        w("hello", 0.0, 0.4), w("there", 0.5, 0.9),       # gap 1.1 s -> break
        w("I", 2.0, 2.1), w("see", 2.2, 2.4), w("you.", 2.5, 2.8),  # punctuation -> break
        w("Again", 2.9, 3.2),
    ]
    lines = group_lines(words)
    assert [line.text for line in lines] == ["hello there", "I see you.", "Again"]
    assert len(group_lines([w(str(i), i * 0.1, i * 0.1 + 0.05) for i in range(25)], max_words=10)) == 3


def test_format_timestamp():
    assert format_timestamp(0) == "00:00.00"
    assert format_timestamp(83.456) == "01:23.46"
    assert format_timestamp(6000) == "100:00.00"


def test_lrc_and_json():
    lines = group_lines([w("la", 1.0, 1.2), w("di", 1.3, 1.5), w("da", 3.0, 3.4)])
    plain = to_lrc(lines, title="Song", duration=125.0)
    assert "[ti:Song]" in plain and "[length:02:05]" in plain
    assert "[00:01.00]la di\n[00:03.00]da\n" in plain

    enhanced = to_lrc(lines, enhanced=True)
    assert "[00:01.00]<00:01.00>la <00:01.30>di <00:01.50>" in enhanced

    doc = to_json(lines, language="en", engine="whistle", duration=125.0)
    assert doc["lines"][0]["words"][1] == {"word": "di", "start": 1.3, "end": 1.5, "probability": 0.9}
