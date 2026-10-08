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
VOX2LRC_API_TOKEN=... uv run uvicorn vox2lrc.server:app --port 8000
```

```sh
curl -H "Authorization: Bearer $TOKEN" -F file=@vocals.mp3 -F language=en -F title="Song" \
  https://<host>/v1/transcribe
```

Returns `{"lrc": "...", "lrc_enhanced": "...", "timed": {...lyrics_timed.json...}}`.
Errors: 401 bad token, 413 upload too large, 422 undecodable/too long/bad language, 429 busy.
Uploads are deleted when the request ends. Settings are in [.env.example](.env.example).

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
