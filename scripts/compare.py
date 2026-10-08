"""Compare engines on a folder of vocal stems.

    uv run python scripts/compare.py STEMS_DIR OUT_DIR --language en \
        --engine whistle --engine faster-whisper:small --engine faster-whisper:medium

Each (stem, engine) runs in its own process, so wall time and peak memory
(max RSS) are measured per run. If STEMS_DIR holds a reference transcript next to
a stem (song.wav -> song.txt, the corrected lyrics), the run is scored by word
edits needed to reach it: substitutions + insertions + deletions, case and
punctuation ignored. Results go to OUT_DIR/results.csv and are printed as a table.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


def normalize(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower().replace("’", "'"))


def word_edits(hyp: list[str], ref: list[str]) -> int:
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        cur = [i] + [0] * len(ref)
        for j, r in enumerate(ref, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (h != r))
        prev = cur
    return prev[-1]


def run(stem: Path, engine: str, out_dir: Path, language: str | None) -> dict:
    tag = engine.replace(":", "-")
    lrc, js = out_dir / f"{stem.stem}.{tag}.lrc", out_dir / f"{stem.stem}.{tag}.json"
    cmd = [sys.executable, "-m", "vox2lrc.cli", str(stem), "--engine", engine, "-o", str(lrc), "--json", str(js)]
    if language:
        cmd += ["--language", language]
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    _, status, usage = os.wait4(proc.pid, 0)
    elapsed = time.monotonic() - t0
    stderr = proc.stderr.read() if proc.stderr else ""
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    rss_mb = usage.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    row = {"stem": stem.name, "engine": engine, "seconds": round(elapsed, 1), "max_rss_mb": round(rss_mb)}
    if os.waitstatus_to_exitcode(status) != 0:
        row["error"] = stderr.strip().splitlines()[-1] if stderr.strip() else "failed"
        return row

    doc = json.loads(js.read_text())
    hyp = normalize(" ".join(line["text"] for line in doc["lines"]))
    row["words"] = len(hyp)
    row["duration"] = doc["duration"]
    row["rtf"] = round(elapsed / doc["duration"], 2) if doc["duration"] else None
    ref_path = stem.with_suffix(".txt")
    if ref_path.exists():
        ref = normalize(ref_path.read_text())
        edits = word_edits(hyp, ref)
        row["ref_words"] = len(ref)
        row["edits"] = edits
        row["wer"] = round(edits / max(len(ref), 1), 3)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stems", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--engine", action="append", dest="engines", help="repeatable; default: whistle")
    p.add_argument("-l", "--language")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(f for f in args.stems.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    if not stems:
        p.error(f"no audio files in {args.stems}")
    engines = args.engines or ["whistle"]

    rows = []
    for stem in stems:
        for engine in engines:
            row = run(stem, engine, args.out, args.language)
            rows.append(row)
            print(json.dumps(row), file=sys.stderr)

    fields = ["stem", "engine", "duration", "seconds", "rtf", "max_rss_mb", "words", "ref_words", "edits", "wer", "error"]
    with open(args.out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nengine                     runs  edits/ref   WER    max RSS   RTF")
    for engine in engines:
        mine = [r for r in rows if r["engine"] == engine and "error" not in r]
        scored = [r for r in mine if "edits" in r]
        edits, ref = sum(r["edits"] for r in scored), sum(r["ref_words"] for r in scored)
        wer = f"{edits / ref:.3f}" if ref else "  -  "
        rss = max((r["max_rss_mb"] for r in mine), default=0)
        rtf = sum(r["rtf"] or 0 for r in mine) / max(len(mine), 1)
        print(f"{engine:<26} {len(mine):>4}  {edits:>5}/{ref:<5} {wer}  {rss:>5} MB  {rtf:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
