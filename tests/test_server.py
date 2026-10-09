import hashlib
import hmac
import json
import shutil
import subprocess
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vox2lrc.server import create_app  # noqa: E402
from vox2lrc.types import Word  # noqa: E402

TOKEN = "test-token"
SECRET = "callback-secret"
AUDIO_URL = "https://acct.r2.cloudflarestorage.com/bucket/songs/1/vocals.mp3?X-Amz-Signature=abc"
CALLBACK = "https://open.example.com/api/internal/transcriptions"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeEngine:
    name = "fake"

    def transcribe(self, pcm, language, hints=None):
        return [Word("hello", 0.2, 0.6, 0.9), Word("world", 0.7, 1.1, 0.8)]


class Calls:
    def __init__(self, source=None):
        self.source = source
        self.fetched = []
        self.callbacks = []

    def fetch(self, url, path, limit):
        self.fetched.append(url)
        if self.source is None:
            raise ValueError("audio file is empty")
        shutil.copyfile(self.source, path)

    def notify(self, url, payload, secret):
        # Sign exactly as the real sender does, so the test checks the receiver's recipe.
        body = json.dumps(payload, ensure_ascii=False).encode()
        ts = str(int(time.time()))
        sig = "sha256=" + hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
        self.callbacks.append((url, payload, ts, sig))


def make_client(calls, **kw):
    app = create_app(FakeEngine, token=TOKEN, callback_secret=SECRET, max_upload_mb=1,
                     audio_hosts=[".r2.cloudflarestorage.com"], callback_hosts=["open.example.com"],
                     fetch=calls.fetch, notify=calls.notify, callback_delays=(0,), **kw)
    return TestClient(app)


@pytest.fixture
def tone_path(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "tone.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", str(path)], check=True)
    return path


@pytest.fixture
def tone(tone_path):
    return tone_path.read_bytes()


@pytest.fixture
def client():
    with make_client(Calls()) as c:
        yield c


def wait_done(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in ("succeeded", "failed") and (job["callback_delivered"] or "error" in job):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_healthz_needs_no_auth(client):
    assert client.get("/healthz").json()["ok"] is True


def test_rejects_missing_or_wrong_token(client):
    files = {"file": ("a.mp3", b"x")}
    assert client.post("/v1/transcribe", files=files).status_code == 401
    assert client.post("/v1/transcribe", files=files, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_transcribe(client, tone):
    r = client.post("/v1/transcribe", headers=AUTH, files={"file": ("tone.mp3", tone)},
                    data={"language": "en", "title": "Tone"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "[ti:Tone]" in body["lrc"] and "hello world" in body["lrc"]
    assert "<00:00.20>hello <00:00.70>world" in body["lrc_enhanced"]
    assert body["timed"]["lines"][0]["words"][0]["word"] == "hello"
    assert body["timed"]["language"] == "en"


def test_rejects_bad_input(client):
    assert client.post("/v1/transcribe", headers=AUTH, files={"file": ("a.mp3", b"not audio")}).status_code == 422
    assert client.post("/v1/transcribe", headers=AUTH, files={"file": ("a.mp3", b"")}).status_code == 422
    big = b"\0" * (1024 * 1024 + 1)
    assert client.post("/v1/transcribe", headers=AUTH, files={"file": ("a.mp3", big)}).status_code == 413
    r = client.post("/v1/transcribe", headers=AUTH, files={"file": ("a.mp3", b"x")}, data={"language": "xx"})
    assert r.status_code == 422


def job_body(**kw):
    return {"id": "job-1", "audio_url": AUDIO_URL, "callback_url": CALLBACK, "language": "en", **kw}


def test_job_runs_and_calls_back(tone_path):
    calls = Calls(tone_path)
    with make_client(calls) as client:
        r = client.post("/v1/jobs", headers=AUTH, json=job_body(title="Tone"))
        assert r.status_code == 202 and r.json()["status"] in ("queued", "running")
        job = wait_done(client, "job-1")
    assert job["status"] == "succeeded"
    assert "hello world" in job["result"]["lrc"]
    assert calls.fetched == [AUDIO_URL]
    [(url, payload, ts, sig)] = calls.callbacks
    assert url == CALLBACK
    assert payload["id"] == "job-1" and payload["status"] == "succeeded"
    assert payload["result"]["timed"]["lines"][0]["text"] == "hello world"
    assert "[ti:Tone]" in payload["result"]["lrc"]


def test_job_resubmit_is_idempotent(tone_path):
    calls = Calls(tone_path)
    with make_client(calls) as client:
        client.post("/v1/jobs", headers=AUTH, json=job_body())
        wait_done(client, "job-1")
        r = client.post("/v1/jobs", headers=AUTH, json=job_body())
        assert r.status_code == 202 and r.json()["status"] == "succeeded"
    assert len(calls.fetched) == 1 and len(calls.callbacks) == 1


def test_failed_job_reports_error_and_can_be_retried():
    calls = Calls(None)
    with make_client(calls) as client:
        client.post("/v1/jobs", headers=AUTH, json=job_body())
        job = wait_done(client, "job-1")
        assert job["status"] == "failed" and job["error"] == "audio file is empty"
        assert calls.callbacks[0][1] == {"id": "job-1", "status": "failed", "error": "audio file is empty"}
        client.post("/v1/jobs", headers=AUTH, json=job_body())
        wait_done(client, "job-1")
    assert len(calls.fetched) == 2


@pytest.mark.parametrize("field,url", [
    ("audio_url", "http://acct.r2.cloudflarestorage.com/x.mp3"),
    ("audio_url", "https://169.254.169.254/latest/meta-data"),
    ("audio_url", "https://r2.cloudflarestorage.com.evil.com/x.mp3"),
    ("callback_url", "https://evil.example.com/hook"),
])
def test_job_rejects_disallowed_urls(client, field, url):
    r = client.post("/v1/jobs", headers=AUTH, json=job_body(**{field: url}))
    assert r.status_code == 422


def test_job_requires_auth_and_valid_id(client):
    assert client.post("/v1/jobs", json=job_body()).status_code == 401
    assert client.post("/v1/jobs", headers=AUTH, json=job_body(id="../etc")).status_code == 422
    assert client.get("/v1/jobs/nope", headers=AUTH).status_code == 404


def test_job_with_lyrics_uses_their_words_and_lines(tone_path):
    calls = Calls(tone_path)
    with make_client(calls) as client:
        client.post("/v1/jobs", headers=AUTH, json=job_body(lyrics="Hello,\nworld!"))
        job = wait_done(client, "job-1")
    timed = job["result"]["timed"]
    assert [line["text"] for line in timed["lines"]] == ["Hello,", "world!"]
    assert timed["lines"][1]["words"][0]["start"] == 0.7  # timing from the transcription
    assert timed["alignment"] == {"used": True, "matched": 2, "total": 2, "ratio": 1.0}
    assert "[00:00.20]Hello,\n[00:00.70]world!" in job["result"]["lrc"]


def test_job_with_unrelated_lyrics_falls_back_to_transcription(tone_path):
    calls = Calls(tone_path)
    with make_client(calls) as client:
        client.post("/v1/jobs", headers=AUTH, json=job_body(lyrics="something else entirely\nabout the sea"))
        job = wait_done(client, "job-1")
    timed = job["result"]["timed"]
    assert timed["lines"][0]["text"] == "hello world"
    assert timed["alignment"]["used"] is False


def test_transcribe_accepts_lyrics(client, tone):
    r = client.post("/v1/transcribe", headers=AUTH, files={"file": ("tone.mp3", tone)},
                    data={"lyrics": "HELLO world"})
    assert r.status_code == 200, r.text
    assert r.json()["timed"]["lines"][0]["text"] == "HELLO world"
