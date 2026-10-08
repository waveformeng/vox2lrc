"""Output formats.

lyrics_timed.json is the source of truth: it keeps each word's start *and* end,
which karaoke highlighting needs. .lrc is derived from it and drops end times.
"""

from importlib.metadata import PackageNotFoundError, version

from .lines import Line

JSON_SCHEMA_VERSION = 1


def _tool_version() -> str:
    try:
        return version("vox2lrc")
    except PackageNotFoundError:
        return "dev"


def format_timestamp(seconds: float) -> str:
    """mm:ss.xx, the LRC timestamp form. Minutes may exceed 59."""
    cs = max(0, round(seconds * 100))
    minutes, cs = divmod(cs, 6000)
    return f"{minutes:02d}:{cs // 100:02d}.{cs % 100:02d}"


def to_lrc(lines: list[Line], *, enhanced: bool = False, title: str | None = None,
           artist: str | None = None, duration: float | None = None) -> str:
    """Line-level LRC, or enhanced LRC (A2 extension) with a <mm:ss.xx> tag before each word."""
    out: list[str] = []
    if title:
        out.append(f"[ti:{title}]")
    if artist:
        out.append(f"[ar:{artist}]")
    if duration is not None:
        out.append(f"[length:{format_timestamp(duration).split(".")[0]}]")
    out.append(f"[re:vox2lrc {_tool_version()}]")
    for line in lines:
        if enhanced:
            body = " ".join(f"<{format_timestamp(w.start)}>{w.word}" for w in line.words)
            body += f" <{format_timestamp(line.end)}>"
        else:
            body = line.text
        out.append(f"[{format_timestamp(line.start)}]{body}")
    return "\n".join(out) + "\n"


def to_json(lines: list[Line], *, language: str | None, engine: str, duration: float | None) -> dict:
    """The lyrics_timed.json document."""
    return {
        "version": JSON_SCHEMA_VERSION,
        "generator": f"vox2lrc {_tool_version()}",
        "engine": engine,
        "language": language,
        "duration": duration,
        "lines": [
            {
                "start": round(line.start, 3),
                "end": round(line.end, 3),
                "text": line.text,
                "words": [
                    {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3),
                     "probability": round(w.probability, 3)}
                    for w in line.words
                ],
            }
            for line in lines
        ],
    }
