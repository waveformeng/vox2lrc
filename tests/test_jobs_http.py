"""The real download / callback code against a local HTTP server."""

import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vox2lrc.jobs import download, post_callback

received = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/")
            self.end_headers()
            return
        body = b"x" * (2000 if self.path == "/big" else 100)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        received.append((self.headers, body))
        self.send_response(204)
        self.end_headers()


@pytest.fixture(scope="module")
def base():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_download(base, tmp_path):
    path = tmp_path / "a"
    download(f"{base}/ok", str(path), limit=1000)
    assert path.read_bytes() == b"x" * 100
    with pytest.raises(ValueError, match="over"):
        download(f"{base}/big", str(path), limit=1000)
    with pytest.raises(Exception):
        download(f"{base}/redirect", str(path), limit=1000)  # redirects are refused


def test_callback_signature(base):
    post_callback(f"{base}/hook", {"id": "j", "status": "succeeded", "result": {"lrc": "[00:01.00]é"}}, "s3cret")
    headers, body = received[-1]
    ts = headers["X-Vox2lrc-Timestamp"]
    expected = "sha256=" + hmac.new(b"s3cret", ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    assert headers["X-Vox2lrc-Signature"] == expected
    assert json.loads(body)["result"]["lrc"] == "[00:01.00]é"


def test_reconcile_ping(base, monkeypatch, capsys):
    from vox2lrc import reconcile

    monkeypatch.setenv("VOX2LRC_RECONCILE_URL", f"{base}/reconcile")
    monkeypatch.setenv("VOX2LRC_CALLBACK_SECRET", "s3cret")
    assert reconcile.main() == 0
    headers, body = received[-1]
    assert body == b"{}"
    ts = headers["X-Vox2lrc-Timestamp"]
    assert headers["X-Vox2lrc-Signature"] == "sha256=" + hmac.new(b"s3cret", ts.encode() + b".{}", hashlib.sha256).hexdigest()

    monkeypatch.delenv("VOX2LRC_RECONCILE_URL")
    assert reconcile.main() == 0  # not configured: does nothing
