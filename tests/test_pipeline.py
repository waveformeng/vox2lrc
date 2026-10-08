"""End-to-end through real ffmpeg with a fake engine. Skipped if ffmpeg is missing."""

import shutil
import subprocess

import pytest

from vox2lrc.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from vox2lrc.pipeline import transcribe_stem
from vox2lrc.types import Word

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


class FakeEngine:
    name = "fake"

    def __init__(self):
        self.calls = []

    def transcribe(self, pcm, language):
        seconds = len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
        self.calls.append(seconds)
        return [Word(f"w{len(self.calls)}", 0.1, min(0.5, seconds), 1.0)]


def test_tone_silence_tone(tmp_path):
    # 3 s tone, 2 s silence, 28 s tone: 0-3 + 5-33 is over 29.5 s, so two chunks with the silence dropped.
    stem = tmp_path / "stem.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=28",
         "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", str(stem)],
        check=True,
    )
    engine = FakeEngine()
    t = transcribe_stem(str(stem), engine, "en")

    assert abs(t.duration - 33) < 0.1
    assert len(engine.calls) == 2 and all(s <= 29.5 for s in engine.calls)
    assert sum(engine.calls) == pytest.approx(31.3, abs=0.3)  # silence skipped, 0.15 s padding kept
    assert t.words[0].start == pytest.approx(0.1, abs=0.05)
    assert t.words[1].start == pytest.approx(t.chunks[1].start + 0.1, abs=0.01)
    assert [w.start for w in t.words] == sorted(w.start for w in t.words)
