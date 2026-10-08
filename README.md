# vox2lrc

Vocals conversion to LRC: transcribe an isolated vocal stem into timed lyrics.
There is no lyrics input; this is transcription, not forced alignment.

Output:
- `.lrc` with line timestamps, or enhanced LRC with a `<mm:ss.xx>` tag per word
- `lyrics_timed.json` with each word's start, end and probability (the format to store; `.lrc` drops word end times)

The default engine is [Whistle](https://huggingface.co/Cactus-Compute/whistle) (Apache-2.0, 16.9 MB, CPU-only,
en/de/fr/es/it/nl/pl). `faster-whisper` is available as an alternative. Requires `ffmpeg` on `PATH`.

## CLI

```sh
uv sync
uv run vox2lrc vocals.mp3 -l en --enhanced -v
```

## HTTP API

```sh
uv sync --extra server
VOX2LRC_API_TOKEN=... VOX2LRC_CALLBACK_SECRET=... uv run uvicorn vox2lrc.server:app --port 8000
```

Transcription takes roughly as long as the song on a 1 vCPU Droplet, so apps use the asynchronous job API:

```
POST /v1/jobs   Authorization: Bearer $VOX2LRC_API_TOKEN
{"id": "<your job id>", "audio_url": "<presigned R2 GET>", "callback_url": "https://<app>/...", "language": "en"}
→ 202 {"id", "status": "queued"}
```

When the job finishes, vox2lrc POSTs `{"id", "status": "succeeded", "result": {"lrc", "lrc_enhanced", "timed"}}`
(or `"status": "failed", "error"`) to `callback_url`, with `X-Vox2lrc-Timestamp` and
`X-Vox2lrc-Signature: sha256=HMAC-SHA256(VOX2LRC_CALLBACK_SECRET, "<timestamp>.<body>")`.
`GET /v1/jobs/{id}` returns the same result for an hour. Jobs are held in memory: if the server restarts, resubmit
any job that never called back (resubmitting a known id is a no-op). `audio_url` and `callback_url` must be https
and on the hosts allowed by `VOX2LRC_AUDIO_HOSTS` / `VOX2LRC_CALLBACK_HOSTS`.

For callers that can't schedule their own recovery, `vox2lrc-reconcile.timer` POSTs a signed `{}` to
`VOX2LRC_RECONCILE_URL` every 5 minutes (`python -m vox2lrc.reconcile`), so the app can resubmit lost jobs then.

`POST /v1/transcribe` (multipart `file=`) is a synchronous version for manual testing. Settings: [.env.example](.env.example).

## Deploy (DigitalOcean Droplet)

On a Droplet prepared by `droplet-init.sh` (ffmpeg, uv, `worker` user), with a DNS A record for the domain:

```sh
deploy/deploy.sh <droplet-ip> <domain>
```

This installs Caddy (HTTPS), opens ports 80/443, runs the API as a hardened systemd service under `worker`,
and on first deploy generates the API token on the server and prints it once.

## Comparing engines

Put stems in `stems/` (gitignored), optionally with the correct lyrics as `<name>.txt`, then:

```sh
uv run python scripts/compare.py stems stems/out -l en --engine whistle --engine faster-whisper:small
```
