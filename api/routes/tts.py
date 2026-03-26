"""
Omnex — Voice Routes
POST /voice/speak       { text, voice?, engine? }  → audio/wav (streaming)
GET  /voice/info                                   → engine info
POST /voice/transcribe  multipart audio file       → { text, language, duration }
"""

from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class SpeakRequest(BaseModel):
    text:   str
    voice:  str | None = None
    engine: str | None = None   # "chatterbox" | "kokoro" | None = use server default


@router.post("/speak")
async def speak(req: SpeakRequest):
    from api.tts import synthesize_stream

    loop = asyncio.get_event_loop()

    async def _async_gen():
        q: queue.Queue = queue.Queue()
        _DONE = object()

        def _produce():
            try:
                for chunk in synthesize_stream(req.text, req.voice, req.engine):
                    q.put(chunk)
            except Exception as e:
                import logging
                logging.getLogger("omnex.tts").error(f"TTS error: {e}")
            finally:
                q.put(_DONE)

        threading.Thread(target=_produce, daemon=True).start()

        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is _DONE:
                break
            yield chunk

    return StreamingResponse(
        _async_gen(),
        media_type="audio/wav",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


@router.get("/info")
async def tts_info():
    from api.tts import engine_info
    return engine_info()


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribe an audio file using local Whisper.
    Accepts any format Whisper/ffmpeg supports (webm, wav, mp3, ogg, …).
    """
    import tempfile
    from embeddings.audio import transcribe as _transcribe

    suffix = ".webm"
    if audio.filename:
        suffix = Path(audio.filename).suffix or ".webm"
    elif audio.content_type:
        ct_map = {
            "audio/webm": ".webm", "audio/wav": ".wav", "audio/mpeg": ".mp3",
            "audio/ogg": ".ogg",  "audio/mp4": ".m4a", "audio/flac": ".flac",
        }
        suffix = ct_map.get(audio.content_type, ".webm")

    data = await audio.read()

    def _run():
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            segments = _transcribe(tmp_path)
            text     = " ".join(s.text for s in segments)
            language = segments[0].language if segments else "unknown"
            duration = segments[-1].end if segments else 0.0
            return {"text": text, "language": language, "duration": round(duration, 2)}
        finally:
            tmp_path.unlink(missing_ok=True)

    return await asyncio.get_event_loop().run_in_executor(None, _run)
