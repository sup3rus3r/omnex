"""
Omnex — Ngrok Tunnel Manager
Auto-opens a public HTTPS tunnel on startup if NGROK_AUTHTOKEN is set.
Uses ngrok CLI directly if available, falls back to pyngrok download.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import logging
import re

log = logging.getLogger("omnex.tunnel")

_url: str | None = None
_error: str | None = None
_status: str = "disabled"
_lock = threading.Lock()


def start_tunnel(port: int = 8000) -> None:
    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        return
    threading.Thread(target=_open_tunnel, args=(token, port), daemon=True, name="omnex-ngrok").start()


def _open_tunnel(token: str, port: int) -> None:
    global _url, _error, _status
    with _lock:
        _status = "starting"

    try:
        ngrok_bin = shutil.which("ngrok")

        if ngrok_bin:
            _tunnel_via_cli(ngrok_bin, token, port)
        else:
            _tunnel_via_pyngrok(token, port)

    except Exception as e:
        with _lock:
            _error  = str(e)
            _status = "error"
        log.error(f"Ngrok tunnel failed: {e}")


def _tunnel_via_cli(ngrok_bin: str, token: str, port: int) -> None:
    """Start ngrok via CLI subprocess and parse the tunnel URL from the API."""
    global _url, _error, _status
    import time, urllib.request, json

    # Configure auth token
    subprocess.run([ngrok_bin, "config", "add-authtoken", token],
                   capture_output=True, timeout=10)

    # Start ngrok in background
    proc = subprocess.Popen(
        [ngrok_bin, "http", str(port), "--log=stdout", "--log-format=json"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Poll ngrok local API for tunnel URL (up to 15s)
    for _ in range(30):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                data = json.loads(r.read())
                tunnels = data.get("tunnels", [])
                for t in tunnels:
                    if t.get("proto") == "https":
                        public_url = t["public_url"]
                        with _lock:
                            _url    = public_url
                            _status = "active"
                            _error  = None
                        log.info(f"Ngrok tunnel active: {public_url}")
                        return
        except Exception:
            pass

    raise RuntimeError("Ngrok started but no tunnel URL found after 15s")


def _tunnel_via_pyngrok(token: str, port: int) -> None:
    """Fall back to pyngrok (downloads binary itself)."""
    global _url, _error, _status
    from pyngrok import ngrok, conf
    cfg = conf.get_default()
    cfg.auth_token = token
    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url.replace("http://", "https://")
    with _lock:
        _url    = public_url
        _status = "active"
        _error  = None
    log.info(f"Ngrok tunnel active (pyngrok): {public_url}")


def get_status() -> dict:
    from api.auth import api_key_enabled
    with _lock:
        return {
            "status":       _status,
            "url":          _url,
            "error":        _error,
            "auth_enabled": api_key_enabled(),
            "api_key":      os.getenv("OMNEX_API_KEY") or None,
        }
