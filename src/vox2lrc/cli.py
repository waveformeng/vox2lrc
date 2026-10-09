import argparse
import json
import subprocess
import sys
from pathlib import Path

from .engines import load_engine
from .lrc import to_json, to_lrc
from .pipeline import transcribe_stem

LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vox2lrc", description="Transcribe an isolated vocal stem into timed lyrics.")
    p.add_argument("input", help="vocal stem (any format ffmpeg reads)")
    p.add_argument("-o", "--lrc", type=Path, help="write .lrc here (default: <input>.lrc)")
    p.add_argument("--json", type=Path, help="write word-timing JSON here (default: <input>.lyrics_timed.json)")
    p.add_argument("-l", "--language", choices=LANGUAGES, help="force the language (recommended)")
    p.add_argument("--lyrics", type=Path,
                   help="text file of the known lyrics, one sung line per line: output uses these words, timed from the audio")
    p.add_argument("--engine", default="whistle", help="whistle (default) or faster-whisper[:small|medium]")
    p.add_argument("--enhanced", action="store_true", help="per-word <mm:ss.xx> tags in the .lrc")
    p.add_argument("--title")
    p.add_argument("--artist")
    p.add_argument("--max-gap", type=float, default=0.6, help="pause in seconds that starts a new line")
    p.add_argument("--max-words", type=int, default=10, help="maximum words per line")
    p.add_argument("--noise-db", type=float, default=-35.0, help="silencedetect threshold for chunking")
    p.add_argument("--max-duration", type=float, default=15 * 60, help="reject longer inputs (seconds)")
    p.add_argument("-v", "--verbose", action="store_true", help="print each chunk as it is transcribed")
    args = p.parse_args(argv)

    stem = Path(args.input)
    if not stem.is_file():
        p.error(f"no such file: {stem}")
    lrc_path = args.lrc or stem.with_suffix(".lrc")
    json_path = args.json or stem.with_suffix(".lyrics_timed.json")

    def on_chunk(chunk, words):
        if args.verbose:
            text = " ".join(w.word for w in words) or "(no speech)"
            print(f"[{chunk.start:7.2f} - {chunk.end:7.2f}] {text}", file=sys.stderr)

    lyrics = args.lyrics.read_text(encoding="utf-8") if args.lyrics else None
    engine = load_engine(args.engine)
    try:
        transcript = transcribe_stem(str(stem), engine, args.language, lyrics=lyrics,
                                     max_duration=args.max_duration, noise_db=args.noise_db, on_chunk=on_chunk)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip().splitlines()
        print(f"vox2lrc: {e.cmd[0]} failed: {detail[-1] if detail else f'exit {e.returncode}'}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"vox2lrc: {e}", file=sys.stderr)
        return 1
    lines = transcript.lines(max_gap=args.max_gap, max_words=args.max_words)

    lrc_path.write_text(to_lrc(lines, enhanced=args.enhanced, title=args.title, artist=args.artist,
                               duration=transcript.duration), encoding="utf-8")
    doc = to_json(lines, language=transcript.language, engine=transcript.engine, duration=transcript.duration,
                  alignment=transcript.alignment_report())
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(transcript.words)} words, {len(lines)} lines, {len(transcript.chunks)} chunks "
          f"-> {lrc_path}, {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
