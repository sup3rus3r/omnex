"""
Omnex — Ngrok Tunnel Manager
Auto-opens a public HTTPS tunnel on startup if NGROK_AUTHTOKEN is set.
The public URL is stored in module state and exposed via /setup/tunnel.
"""

from __future__ import annotations

import os
import threading
import logging

log = logging.getLogger("omnex.tunnel")

_url: str | None = None
_error: str | None = None
_status: str = "disabled"   # disabled | starting | active | error
_lock = threading.Lock()


def start_tunnel(port: int = 8000) -> None:
    """
    Start ngrok tunnel in a background thread.
    Does nothing if NGROK_AUTHTOKEN is not set.
    """
    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        return

    threading.Thread(target=_open_tunnel, args=(token, port), daemon=True, name="omnex-ngrok").start()


def _open_tunnel(token: str, port: int) -> None:
    global _url, _error, _status
    with _lock:
        _status = "starting"

    try:
        from pyngrok import ngrok, conf
        cfg = conf.get_default()
        cfg.auth_token = token
        # Use pre-installed binary if available (avoids SSL download issues in Docker)
        import shutil
        system_ngrok = shutil.which("ngrok")
        if system_ngrok:
            cfg.ngrok_path = system_ngrok
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url.replace("http://", "https://")

        with _lock:
            _url    = public_url
            _status = "active"
            _error  = None

        log.info(f"Ngrok tunnel active: {public_url}")

    except Exception as e:
        with _lock:
            _error  = str(e)
            _status = "error"
        log.error(f"Ngrok tunnel failed: {e}")


def get_status() -> dict:
    import os
    from api.auth import api_key_enabled
    with _lock:
        return {
            "status":       _status,
            "url":          _url,
            "error":        _error,
            "auth_enabled": api_key_enabled(),
            "api_key":      os.getenv("OMNEX_API_KEY") or None,
        }
