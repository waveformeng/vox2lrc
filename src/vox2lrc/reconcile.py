"""Ask the app to reconcile its transcription jobs with this server.

Run every few minutes by deploy/vox2lrc-reconcile.timer, for apps that can't
schedule their own (Waveform's Vercel plan has no cron jobs). POSTs {} to
VOX2LRC_RECONCILE_URL, signed like callbacks with VOX2LRC_CALLBACK_SECRET.
The app then resubmits jobs this server lost (they're held in memory) and
fetches results whose callback never arrived.
"""

import os
import sys
import urllib.error

from .jobs import post_callback


def main() -> int:
    url = os.environ.get("VOX2LRC_RECONCILE_URL", "")
    secret = os.environ.get("VOX2LRC_CALLBACK_SECRET", "")
    if not url:
        print("VOX2LRC_RECONCILE_URL is not set; nothing to do", file=sys.stderr)
        return 0
    if not secret:
        print("VOX2LRC_CALLBACK_SECRET is not set", file=sys.stderr)
        return 1
    try:
        body = post_callback(url, {}, secret, timeout=90)
    except urllib.error.HTTPError as e:
        print(f"reconcile failed: HTTP {e.code}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"reconcile failed: {e}", file=sys.stderr)
        return 1
    print(body.decode("utf-8", "replace") if body else "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
