import shutil
import subprocess

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vox2lrc.server import create_app  # noqa: E402
from vox2lrc.types import Word  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeEngine:
    name = "fake"

    def transcribe(self, pcm, language):
        return [Word("hello", 0.2, 0.6, 0.9), Word("world", 0.7, 1.1, 0.8)]


@pytest.fixture
def client():
    with TestClient(create_app(FakeEngine, token=TOKEN, max_upload_mb=1)) as c:
        yield c


@pytest.fixture
def tone(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "tone.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", str(path)], check=True)
    return path.read_bytes()


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
