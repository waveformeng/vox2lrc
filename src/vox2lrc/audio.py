"""Audio handling via ffmpeg subprocesses.

Inputs are untrusted uploads, so ffmpeg/ffprobe are restricted to local files
and never read stdin. Audio is streamed as 16 kHz mono float32 PCM and read one
chunk at a time; the whole stem is never held in memory.
"""

import json
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 4  # float32
MAX_CHUNK_SECONDS = 29.5  # Whistle accepts at most 30 s per call

_FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-protocol_whitelist", "file"]
_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def detect_silences(path: str, duration: float, noise_db: float = -35.0,
                    min_silence: float = 0.3) -> list[Span]:
    """Silent regions of the stem, from ffmpeg's silencedetect filter."""
    proc = subprocess.run(
        [*_FFMPEG, "-loglevel", "info", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    return parse_silencedetect(proc.stderr, duration)


def parse_silencedetect(stderr: str, duration: float | None = None) -> list[Span]:
    silences: list[Span] = []
    start: float | None = None
    for line in stderr.splitlines():
        if m := _SILENCE_START.search(line):
            start = max(0.0, float(m.group(1)))
        elif (m := _SILENCE_END.search(line)) and start is not None:
            silences.append(Span(start, float(m.group(1))))
            start = None
    if start is not None and duration is not None:  # trailing silence runs to the end
        silences.append(Span(start, duration))
    return silences


def plan_chunks(duration: float, silences: list[Span], max_len: float = MAX_CHUNK_SECONDS,
                pad: float = 0.15) -> list[Span]:
    """Group the voiced regions between silences into chunks no longer than max_len.

    Silent stretches are dropped. A voiced region longer than max_len is cut into
    equal pieces; those hard cuts may split a word.
    """
    voiced: list[Span] = []
    cursor = 0.0
    for s in sorted(silences, key=lambda s: s.start):
        if s.start > cursor:
            voiced.append(Span(cursor, s.start))
        cursor = max(cursor, s.end)
    if cursor < duration:
        voiced.append(Span(cursor, duration))

    pieces: list[Span] = []
    for v in voiced:
        if v.duration <= max_len:
            pieces.append(v)
            continue
        n = int(v.duration // max_len) + 1
        step = v.duration / n
        pieces.extend(Span(v.start + i * step, v.start + (i + 1) * step) for i in range(n))

    chunks: list[Span] = []
    for p in pieces:
        if chunks and p.end - chunks[-1].start <= max_len:
            chunks[-1] = Span(chunks[-1].start, p.end)
        else:
            chunks.append(p)

    # Pad into the surrounding silence without overlapping neighbours or exceeding max_len.
    padded: list[Span] = []
    for i, c in enumerate(chunks):
        lo = padded[-1].end if padded else 0.0
        hi = chunks[i + 1].start if i + 1 < len(chunks) else duration
        start = max(lo, c.start - pad)
        end = min(hi, c.end + pad, start + max_len)
        padded.append(Span(start, end))
    return padded


def iter_chunk_pcm(path: str, chunks: list[Span]) -> Iterator[tuple[Span, bytes]]:
    """Decode the stem once and yield each chunk as 16 kHz mono float32 PCM bytes."""
    proc = subprocess.Popen(
        [*_FFMPEG, "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    position = 0  # samples consumed so far
    try:
        for chunk in chunks:
            first = round(chunk.start * SAMPLE_RATE)
            last = round(chunk.end * SAMPLE_RATE)
            _skip(proc.stdout, (first - position) * BYTES_PER_SAMPLE)
            data = _read(proc.stdout, (last - first) * BYTES_PER_SAMPLE)
            position = last
            if data:
                yield chunk, data
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _skip(stream, n: int) -> None:
    while n > 0:
        got = stream.read(min(n, 1 << 20))
        if not got:
            return
        n -= len(got)


def _read(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        got = stream.read(n - len(buf))
        if not got:
            break
        buf += got
    return bytes(buf[: len(buf) - len(buf) % BYTES_PER_SAMPLE])
