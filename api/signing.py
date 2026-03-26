"""
Omnex — URL Signing
HMAC-SHA256 signed, time-limited URLs for media endpoints.

Usage:
  from api.signing import sign_url, verify_signed_request

  # Generate a signed URL (expires in 3600 seconds by default)
  url = sign_url("/chunk/abc123/raw")

  # Verify in a route handler (raises HTTPException on failure)
  verify_signed_request(token, expires, chunk_id)

Set OMNEX_MEDIA_SECRET env var to enable signing.
If unset, signing is disabled (open access — suitable for local-only use).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time

from fastapi import HTTPException


def _secret() -> bytes | None:
    s = os.getenv("OMNEX_MEDIA_SECRET")
    return s.encode() if s else None


def signing_enabled() -> bool:
    return _secret() is not None


def sign_url(path: str, expires_in: int = 3600) -> str:
    """
    Return a signed URL for the given path.
    If signing is disabled, returns path unchanged.
    """
    secret = _secret()
    if not secret:
        return path

    expires = int(time.time()) + expires_in
    token   = _make_token(secret, path, expires)
    return f"{path}?token={token}&expires={expires}"


def verify_signed_request(token: str | None, expires: int | None, path: str) -> None:
    """
    Verify a signed request. Raises HTTPException on failure.
    If signing is disabled, always passes.
    """
    secret = _secret()
    if not secret:
        return  # Signing disabled — open access

    if not token or not expires:
        raise HTTPException(status_code=403, detail="Missing signature")

    if int(time.time()) > int(expires):
        raise HTTPException(status_code=403, detail="URL expired")

    expected = _make_token(secret, path, int(expires))
    if not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="Invalid signature")


def _make_token(secret: bytes, path: str, expires: int) -> str:
    msg = f"{path}:{expires}".encode()
    return hmac.HMAC(key=secret, msg=msg, digestmod=hashlib.sha256).hexdigest()
