"""
Omnex — API Key Authentication
Gates protected routes with X-API-Key header.

Set OMNEX_API_KEY env var to enable. If unset, auth is disabled (local-only mode).
"""

from __future__ import annotations

import os
from fastapi import Header, HTTPException, Request
from fastapi.security import APIKeyHeader

_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _expected_key() -> str | None:
    return os.getenv("OMNEX_API_KEY") or None


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    FastAPI dependency — call as Depends(require_api_key).
    Passes if:
      - OMNEX_API_KEY is not set (local-only mode, no auth)
      - X-API-Key header matches OMNEX_API_KEY
    Raises 401 otherwise.
    """
    expected = _expected_key()
    if expected is None:
        return  # Auth disabled
    if x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def api_key_enabled() -> bool:
    return _expected_key() is not None
