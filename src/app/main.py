from __future__ import annotations
import json
import uuid
import time
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from loguru import logger

from .version import VERSION
from .models import TranscribeResponse, PipelineInfo, SeparationInfo, TranscriptionInfo, Timings, ConfigInput, Segment
from .audio import read_audio_bytes_to_mono16k, write_wav_bytes
from .separation import separate_vocals_if_available
from .asr import vad_chunks, transcribe_chunks
from .diarize import maybe_diarize

app = FastAPI(title="Voice Separation + Transcription API", version=VERSION)

MAX_BYTES = 100 * 1024 * 1024  # 100 MB

@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}

@app.post("/v1/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio file: wav/mp3/m4a/flac/ogg"),
    config: Optional[str] = Form(None, description="JSON string for optional params"),
):
    req_id = str(uuid.uuid4())
    start_total = time.perf_counter()
    timings = {"load": 0, "separation": 0, "transcription": 0, "total": 0}

    try:
        cfg = ConfigInput(**(json.loads(config) if config else {}))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")

    # Read file
    blob = await file.read()
    if len(blob) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(blob) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        t0 = time.perf_counter()
        audio_f32, sr = read_audio_bytes_to_mono16k(blob, target_sr=cfg.target_sr)
        timings["load"] = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        logger.exception("decode failure")
        raise HTTPException(status_code=422, detail=f"Decode failed: {e}")

    duration_sec = len(audio_f32) / sr

    # Separation
    sep_enabled = bool(cfg.enable_separation)
    sep_method = "disabled"
    fallback_used = False
    try:
        t0 = time.perf_counter()
        if sep_enabled:
            wav_bytes = write_wav_bytes(audio_f32, sr)
            sep_bytes, sep_method, fallback_used = separate_vocals_if_available(wav_bytes)
            # decode back after separation
            audio_f32, sr = read_audio_bytes_to_mono16k(sep_bytes, target_sr=cfg.target_sr)
        timings["separation"] = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        logger.warning(f"Separation failed; continuing with raw audio. err={e}")
        sep_enabled = False
        sep_method = "failed"
        fallback_used = False

    # VAD/chunking (simple pass-through; faster-whisper also has VAD options)
    # For long files you could chunk via VAD first, then transcribe chunk by chunk and stitch timestamps.
    # Here we transcribe in one pass for simplicity and speed. 
    try:
        t0 = time.perf_counter()
        segments, language = transcribe_chunks(audio_f32, sr, cfg.model_size, cfg.language_hint)
        timings["transcription"] = int((time.perf_counter() - t0) * 1000)
    except Exception as e:
        logger.exception("ASR failure")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Optional diarization (placeholder, no-op unless integrated)
    segments = maybe_diarize(segments, enable=cfg.diarize)

    resp = TranscribeResponse(
        request_id=req_id,
        duration_sec=round(duration_sec, 3),
        sample_rate=sr,
        pipeline=PipelineInfo(
            separation=SeparationInfo(enabled=sep_enabled, method=sep_method, fallback_used=fallback_used),
            transcription=TranscriptionInfo(model=f"whisper-{cfg.model_size}")
        ),
        segments=[Segment(**s) for s in segments],
        text=" ".join([s["text"] for s in segments]).strip(),
        language=language,
        timings_ms=Timings(**{**timings, "total": int((time.perf_counter() - start_total) * 1000)}),
    )
    headers = {"X-Request-ID": req_id}
    return JSONResponse(content=json.loads(resp.model_dump_json()), headers=headers)
