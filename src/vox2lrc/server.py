"""HTTP API: upload a vocal stem, get timed lyrics back.

    POST /v1/transcribe   (Authorization: Bearer $VOX2LRC_API_TOKEN)
      multipart form: file=<audio>, language=en (optional), title, artist
      200 -> {"lrc": str, "lrc_enhanced": str, "timed": {...lyrics_timed.json...}}
    GET  /healthz         (no auth)

The upload is written to a private (0700) temp dir and deleted when the request ends;
nothing is kept on the server. One transcription runs at a time (1 vCPU, and
the Whistle engine is not thread-safe); up to VOX2LRC_MAX_QUEUE more wait, the
rest get 429.

Environment:
  VOX2LRC_API_TOKEN      required; shared secret for the Authorization header
  VOX2LRC_ENGINE         whistle (default) or faster-whisper[:model]
  VOX2LRC_MAX_UPLOAD_MB  default 50
  VOX2LRC_MAX_DURATION   seconds of audio, default 900
  VOX2LRC_MAX_QUEUE      requests allowed to wait, default 4

Run: uvicorn vox2lrc.server:app --host 127.0.0.1 --port 8000
"""

import asyncio
import hmac
import os
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .cli import LANGUAGES
from .engines import Engine, load_engine
from .lines import group_lines
from .lrc import to_json, to_lrc
from .pipeline import transcribe_stem

_COPY_BLOCK = 1 << 20


def create_app(engine_factory: Callable[[], Engine] | None = None, *, token: str | None = None,
               max_upload_mb: float | None = None, max_duration: float | None = None,
               max_queue: int | None = None) -> FastAPI:
    token = token if token is not None else os.environ.get("VOX2LRC_API_TOKEN", "")
    if not token:
        raise RuntimeError("VOX2LRC_API_TOKEN is not set")
    max_upload = int((max_upload_mb or float(os.environ.get("VOX2LRC_MAX_UPLOAD_MB", 50))) * 1024 * 1024)
    max_duration = max_duration or float(os.environ.get("VOX2LRC_MAX_DURATION", 900))
    max_queue = max_queue if max_queue is not None else int(os.environ.get("VOX2LRC_MAX_QUEUE", 4))
    engine_factory = engine_factory or (lambda: load_engine(os.environ.get("VOX2LRC_ENGINE", "whistle")))

    state = {"engine": None, "waiting": 0}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        # Load (and on first run download) the model before accepting requests.
        state["engine"] = await run_in_threadpool(engine_factory)
        yield

    app = FastAPI(title="vox2lrc", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        scheme, _, given = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": state["engine"] is not None, "busy": lock.locked(), "waiting": state["waiting"]}

    @app.post("/v1/transcribe", dependencies=[Depends(authorize)])
    async def transcribe(
        file: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = None,
        title: Annotated[str | None, Form()] = None,
        artist: Annotated[str | None, Form()] = None,
    ) -> dict:
        if language is not None and language not in LANGUAGES:
            raise HTTPException(422, f"language must be one of {', '.join(LANGUAGES)}")
        if state["waiting"] >= max_queue and lock.locked():
            raise HTTPException(429, "busy, try again shortly", headers={"Retry-After": "30"})

        with tempfile.TemporaryDirectory(prefix="vox2lrc-") as workdir:
            path = os.path.join(workdir, "input")
            await run_in_threadpool(_save_upload, file, path, max_upload)

            state["waiting"] += 1
            queued = True
            try:
                async with lock:
                    state["waiting"] -= 1
                    queued = False
                    transcript = await run_in_threadpool(
                        transcribe_stem, path, state["engine"], language, max_duration=max_duration)
            except subprocess.CalledProcessError:
                raise HTTPException(422, "could not decode the audio file") from None
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            finally:
                if queued:  # client went away while waiting
                    state["waiting"] -= 1

        lines = group_lines(transcript.words)
        meta = {"title": title, "artist": artist, "duration": transcript.duration}
        return {
            "lrc": to_lrc(lines, **meta),
            "lrc_enhanced": to_lrc(lines, enhanced=True, **meta),
            "timed": to_json(lines, language=transcript.language, engine=transcript.engine,
                             duration=transcript.duration),
        }

    return app


def _save_upload(upload: UploadFile, path: str, limit: int) -> None:
    written = 0
    with open(path, "wb") as out:
        while block := upload.file.read(_COPY_BLOCK):
            written += len(block)
            if written > limit:
                raise HTTPException(413, f"file is over {limit // (1024 * 1024)} MB")
            out.write(block)
    if written == 0:
        raise HTTPException(422, "empty file")


def __getattr__(name: str):
    # `uvicorn vox2lrc.server:app` builds the app from the environment on first access.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
