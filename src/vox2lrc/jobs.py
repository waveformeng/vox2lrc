"""Asynchronous transcription jobs.

The caller submits {id, audio_url, callback_url}; a single background thread
downloads the audio, transcribes it, and POSTs the result to callback_url,
signed with HMAC-SHA256. Jobs live in memory only: a restart forgets them, so
the caller must treat its own job table as the source of truth and resubmit
jobs that never called back. Results are also kept for RESULT_TTL seconds so
the caller can fetch them with GET /v1/jobs/{id} if a callback was lost.

Callback request:
    POST <callback_url>
    Content-Type: application/json
    X-Vox2lrc-Timestamp: <unix seconds>
    X-Vox2lrc-Signature: sha256=<hex HMAC-SHA256(secret, "<timestamp>.<raw body>")>
    {"id": ..., "status": "succeeded", "result": {"lrc", "lrc_enhanced", "timed"}}
    {"id": ..., "status": "failed", "error": "..."}
"""

import hashlib
import hmac
import json
import logging
import os
import queue
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

log = logging.getLogger("vox2lrc.jobs")

RESULT_TTL = 3600
CALLBACK_DELAYS = (0, 2, 5, 15, 30, 60)  # seconds before each callback attempt


class RejectedURL(ValueError):
    pass


@dataclass
class Job:
    id: str
    audio_url: str
    callback_url: str | None
    language: str | None = None
    title: str | None = None
    artist: str | None = None
    status: str = "queued"  # queued | running | succeeded | failed
    error: str | None = None
    result: dict | None = None
    callback_delivered: bool = False
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self, *, with_result: bool = True) -> dict:
        out = {"id": self.id, "status": self.status, "callback_delivered": self.callback_delivered}
        if self.error:
            out["error"] = self.error
        if with_result and self.result is not None:
            out["result"] = self.result
        return out


def check_url(url: str, allowed_hosts: list[str], *, allow_http: bool = False) -> None:
    """Only https URLs whose host matches an allowed entry. An entry starting
    with '.' matches that domain's subdomains (e.g. '.r2.cloudflarestorage.com')."""
    parts = urlsplit(url)
    if parts.scheme != "https" and not (allow_http and parts.scheme == "http"):
        raise RejectedURL(f"URL must be https: {parts.scheme}://{parts.hostname}")
    host = (parts.hostname or "").lower()
    for entry in allowed_hosts:
        entry = entry.strip().lower()
        if entry and (host == entry or (entry.startswith(".") and host.endswith(entry))):
            return
    raise RejectedURL(f"host not allowed: {host}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # a redirect could point anywhere; treat it as an error


_opener = urllib.request.build_opener(_NoRedirect)


def download(url: str, path: str, limit: int, timeout: float = 60) -> None:
    with _opener.open(urllib.request.Request(url, method="GET"), timeout=timeout) as resp, open(path, "wb") as out:
        length = resp.headers.get("Content-Length")
        if length and int(length) > limit:
            raise ValueError(f"audio is over {limit // (1024 * 1024)} MB")
        written = 0
        while block := resp.read(1 << 20):
            written += len(block)
            if written > limit:
                raise ValueError(f"audio is over {limit // (1024 * 1024)} MB")
            out.write(block)
    if written == 0:
        raise ValueError("audio file is empty")


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def post_callback(url: str, payload: dict, secret: str, timeout: float = 30) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode()
    ts = str(int(time.time()))
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "User-Agent": "vox2lrc",
        "X-Vox2lrc-Timestamp": ts,
        "X-Vox2lrc-Signature": sign(secret, ts, body),
    })
    with _opener.open(req, timeout=timeout) as resp:
        resp.read()


class JobRunner:
    """One worker thread; jobs run one at a time in submission order."""

    def __init__(self, process: Callable[[Job, str], dict], *, callback_secret: str, max_queue: int = 20,
                 max_download: int = 50 * 1024 * 1024,
                 fetch: Callable[[str, str, int], None] = download,
                 notify: Callable[[str, dict, str], None] = post_callback,
                 callback_delays: tuple[float, ...] = CALLBACK_DELAYS):
        self._process = process
        self._secret = callback_secret
        self._max_download = max_download
        self._fetch = fetch
        self._notify = notify
        self._delays = callback_delays
        self._queue: queue.Queue[Job] = queue.Queue(maxsize=max_queue)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="vox2lrc-jobs", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, job: Job) -> tuple[Job, bool]:
        """Queue a job. Resubmitting an id that is queued, running or finished
        returns the existing job (idempotent). Raises queue.Full when busy."""
        with self._lock:
            self._prune()
            existing = self._jobs.get(job.id)
            if existing and existing.status != "failed":
                return existing, False
            self._queue.put_nowait(job)  # raises queue.Full before the job is recorded
            self._jobs[job.id] = job
            return job, True

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stats(self) -> dict:
        return {"queued": self._queue.qsize(), "running": any(j.status == "running" for j in self._jobs.values())}

    def _prune(self) -> None:
        cutoff = time.time() - RESULT_TTL
        for jid in [jid for jid, j in self._jobs.items() if j.finished and j.finished < cutoff]:
            del self._jobs[jid]

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            self.run_job(job)

    def run_job(self, job: Job) -> None:
        job.status = "running"
        try:
            with tempfile.TemporaryDirectory(prefix="vox2lrc-job-") as workdir:
                path = os.path.join(workdir, "input")
                self._fetch(job.audio_url, path, self._max_download)
                job.result = self._process(job, path)
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            log.warning("job %s failed: %s", job.id, exc)
            job.error = _describe(exc)
            job.status = "failed"
        job.finished = time.time()
        if job.callback_url:
            self._deliver(job)

    def _deliver(self, job: Job) -> None:
        payload = job.public()
        payload.pop("callback_delivered", None)
        for delay in self._delays:
            time.sleep(delay)
            try:
                self._notify(job.callback_url, payload, self._secret)
                job.callback_delivered = True
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("callback for job %s failed: %s", job.id, exc)


def _describe(exc: Exception) -> str:
    import subprocess

    if isinstance(exc, subprocess.CalledProcessError):
        return "could not decode the audio file"
    if isinstance(exc, urllib.error.HTTPError):
        return f"audio download failed: HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"audio download failed: {exc.reason}"
    if isinstance(exc, ValueError):
        return str(exc)
    return f"internal error: {type(exc).__name__}"
