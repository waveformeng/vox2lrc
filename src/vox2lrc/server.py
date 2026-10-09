"""HTTP API for vox2lrc. Every endpoint except /healthz needs
Authorization: Bearer $VOX2LRC_API_TOKEN.

    POST /v1/jobs          asynchronous (use this from apps)
      JSON {"id": caller's job id, "audio_url": https link to the stem,
            "callback_url": where to POST the result, "language", "title", "artist"}
      202 -> {"id", "status": "queued"}; resubmitting the same id is a no-op.
      When done, the result is POSTed to callback_url, signed (see jobs.py).
    GET  /v1/jobs/{id}     status, and the result once finished (kept 1 hour)

    POST /v1/transcribe    synchronous, for testing: multipart file=<audio>,
      language, title, artist -> {"lrc", "lrc_enhanced", "timed"}. Takes about
      as long as the song on a 1 vCPU Droplet, so most HTTP clients will time out.

    GET  /healthz          no auth

Audio is only ever in a private temp dir for the length of one job. One
transcription runs at a time (1 vCPU, and Whistle is not thread-safe).

Environment:
  VOX2LRC_API_TOKEN        required; bearer token callers send
  VOX2LRC_CALLBACK_SECRET  required; HMAC key for signing callbacks
  VOX2LRC_AUDIO_HOSTS      hosts audio_url may point at, comma-separated; a
                           leading '.' allows subdomains (default .r2.cloudflarestorage.com)
  VOX2LRC_CALLBACK_HOSTS   hosts callback_url may point at (default: none, so callbacks are refused)
  VOX2LRC_ENGINE           whistle (default) or faster-whisper[:model]
  VOX2LRC_MAX_UPLOAD_MB    default 50
  VOX2LRC_MAX_DURATION     seconds of audio, default 900
  VOX2LRC_MAX_QUEUE        jobs allowed to wait, default 20

Run: uvicorn vox2lrc.server:app --host 127.0.0.1 --port 8000
"""

import hmac
import logging
import os
import queue
import subprocess
import tempfile
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .cli import LANGUAGES
from .engines import Engine, load_engine
from .align import MAX_LYRICS_LENGTH
from .jobs import Job, JobRunner, RejectedURL, check_url, download, post_callback
from .lrc import to_json, to_lrc
from .pipeline import transcribe_stem

_COPY_BLOCK = 1 << 20


class JobRequest(BaseModel):
    id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    audio_url: str = Field(max_length=4096)
    callback_url: str | None = Field(default=None, max_length=2048)
    language: str | None = None
    title: str | None = Field(default=None, max_length=300)
    artist: str | None = Field(default=None, max_length=300)
    # Known lyrics, one sung line per line: the output uses these words and lines,
    # timed from the audio.
    lyrics: str | None = Field(default=None, max_length=MAX_LYRICS_LENGTH)


def _env_list(name: str, default: str = "") -> list[str]:
    return [h.strip() for h in os.environ.get(name, default).split(",") if h.strip()]


def create_app(engine_factory: Callable[[], Engine] | None = None, *, token: str | None = None,
               callback_secret: str | None = None, audio_hosts: list[str] | None = None,
               callback_hosts: list[str] | None = None, allow_http: bool = False,
               max_upload_mb: float | None = None, max_duration: float | None = None,
               max_queue: int | None = None, fetch=download, notify=post_callback,
               callback_delays: tuple[float, ...] | None = None) -> FastAPI:
    token = token if token is not None else os.environ.get("VOX2LRC_API_TOKEN", "")
    callback_secret = callback_secret if callback_secret is not None else os.environ.get("VOX2LRC_CALLBACK_SECRET", "")
    if not token or not callback_secret:
        raise RuntimeError("VOX2LRC_API_TOKEN and VOX2LRC_CALLBACK_SECRET must be set")
    audio_hosts = audio_hosts if audio_hosts is not None else _env_list("VOX2LRC_AUDIO_HOSTS", ".r2.cloudflarestorage.com")
    callback_hosts = callback_hosts if callback_hosts is not None else _env_list("VOX2LRC_CALLBACK_HOSTS")
    max_upload = int((max_upload_mb or float(os.environ.get("VOX2LRC_MAX_UPLOAD_MB", 50))) * 1024 * 1024)
    max_duration = max_duration or float(os.environ.get("VOX2LRC_MAX_DURATION", 900))
    max_queue = max_queue if max_queue is not None else int(os.environ.get("VOX2LRC_MAX_QUEUE", 20))
    engine_factory = engine_factory or (lambda: load_engine(os.environ.get("VOX2LRC_ENGINE", "whistle")))

    state: dict = {"engine": None}
    engine_lock = threading.Lock()  # shared by the job thread and /v1/transcribe

    def render(path: str, language: str | None, title: str | None, artist: str | None,
               lyrics: str | None = None) -> dict:
        with engine_lock:
            transcript = transcribe_stem(path, state["engine"], language, lyrics=lyrics or None,
                                         max_duration=max_duration)
        lines = transcript.lines()
        meta = {"title": title, "artist": artist, "duration": transcript.duration}
        return {
            "lrc": to_lrc(lines, **meta),
            "lrc_enhanced": to_lrc(lines, enhanced=True, **meta),
            "timed": to_json(lines, language=transcript.language, engine=transcript.engine,
                             duration=transcript.duration, alignment=transcript.alignment_report()),
        }

    runner_kwargs = {"callback_delays": callback_delays} if callback_delays is not None else {}
    runner = JobRunner(lambda job, path: render(path, job.language, job.title, job.artist, job.lyrics),
                       callback_secret=callback_secret, max_queue=max_queue, max_download=max_upload,
                       fetch=fetch, notify=notify, **runner_kwargs)

    @asynccontextmanager
    async def lifespan(_app):
        # Load (and on first run download) the model before accepting requests.
        state["engine"] = await run_in_threadpool(engine_factory)
        runner.start()
        yield

    app = FastAPI(title="vox2lrc", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runner = runner

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        scheme, _, given = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    def check_language(language: str | None) -> None:
        if language is not None and language not in LANGUAGES:
            raise HTTPException(422, f"language must be one of {', '.join(LANGUAGES)}")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": state["engine"] is not None, **runner.stats()}

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    def create_job(req: JobRequest) -> dict:
        check_language(req.language)
        try:
            check_url(req.audio_url, audio_hosts, allow_http=allow_http)
            if req.callback_url:
                check_url(req.callback_url, callback_hosts, allow_http=allow_http)
        except RejectedURL as e:
            raise HTTPException(422, str(e)) from None
        try:
            job, _ = runner.submit(Job(req.id, req.audio_url, req.callback_url, req.language, req.title, req.artist,
                                         req.lyrics))
        except queue.Full:
            raise HTTPException(429, "queue is full, try again later", headers={"Retry-After": "60"}) from None
        return job.public(with_result=False)

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_job(job_id: str) -> dict:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job (never submitted, expired, or the server restarted)")
        return job.public()

    @app.post("/v1/transcribe", dependencies=[Depends(authorize)])
    async def transcribe(
        file: Annotated[UploadFile, File()],
        language: Annotated[str | None, Form()] = None,
        title: Annotated[str | None, Form()] = None,
        artist: Annotated[str | None, Form()] = None,
        lyrics: Annotated[str | None, Form(max_length=MAX_LYRICS_LENGTH)] = None,
    ) -> dict:
        check_language(language)
        if engine_lock.locked():
            raise HTTPException(429, "busy, try again shortly", headers={"Retry-After": "30"})
        with tempfile.TemporaryDirectory(prefix="vox2lrc-") as workdir:
            path = os.path.join(workdir, "input")
            await run_in_threadpool(_save_upload, file, path, max_upload)
            try:
                return await run_in_threadpool(render, path, language, title, artist, lyrics)
            except subprocess.CalledProcessError:
                raise HTTPException(422, "could not decode the audio file") from None
            except ValueError as e:
                raise HTTPException(422, str(e)) from None

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
        logging.basicConfig(level=logging.INFO)
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
