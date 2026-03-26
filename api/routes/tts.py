"""
Omnex — Voice Routes
POST /voice/speak       { text, voice? }      → audio/wav
GET  /voice/info                              → engine info
POST /voice/transcribe  multipart audio file  → { text, language, duration }
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class SpeakRequest(BaseModel):
    text: str
    voice: str | None = None


@router.post("/speak")
async def speak(req: SpeakRequest):
    import asyncio
    from api.tts import synthesize_stream

    loop = asyncio.get_event_loop()

    def _gen():
        return synthesize_stream(req.text, req.voice)

    # Run the generator in a thread so streaming doesn't block the event loop
    async def _async_gen():
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        gen = await loop.run_in_executor(None, _gen)
        for chunk in gen:
            yield chunk

    return StreamingResponse(
        _async_gen(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
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
    Returns { text, language, duration }.
    """
    import asyncio
    from embeddings.audio import transcribe as _transcribe

    # Determine a sane suffix from content-type or filename
    suffix = ".webm"
    if audio.filename:
        suffix = Path(audio.filename).suffix or ".webm"
    elif audio.content_type:
        ct_map = {
            "audio/webm": ".webm", "audio/wav": ".wav", "audio/mpeg": ".mp3",
            "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/flac": ".flac",
        }
        suffix = ct_map.get(audio.content_type, ".webm")

    data = await audio.read()

    def _run():
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            segments = _transcribe(tmp_path)
            text = " ".join(s.text for s in segments)
            language = segments[0].language if segments else "unknown"
            duration = segments[-1].end if segments else 0.0
            return {"text": text, "language": language, "duration": round(duration, 2)}
        finally:
            tmp_path.unlink(missing_ok=True)

    return await asyncio.get_event_loop().run_in_executor(None, _run)
