from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import webrtcvad
from dataclasses import dataclass
import time

# ASR: faster-whisper
from faster_whisper import WhisperModel

@dataclass
class ASRModels:
    model: Optional[WhisperModel] = None
    model_size: str = "small"
    device: str = "cpu"

_GLOBAL = ASRModels()

def init_model_if_needed(model_size: str = "small") -> ASRModels:
    global _GLOBAL
    if _GLOBAL.model is None or _GLOBAL.model_size != model_size:
        # device auto-detect
        device = "cpu"
        try:
            # ctranslate2 probing: if CUDA available, let faster-whisper use it
            import ctranslate2
            if "cuda" in ctranslate2.get_supported_compute_types():
                device = "cuda"
        except Exception:
            pass
        _GLOBAL = ASRModels(
            model=WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8"),
            model_size=model_size,
            device=device
        )
    return _GLOBAL

def vad_chunks(audio: np.ndarray, sr: int, aggressiveness: int = 2, min_chunk_ms: int = 1000, max_chunk_ms: int = 30000, pad_ms: int = 200) -> List[Tuple[int, int]]:
    """Return list of (start_sample, end_sample) for voiced segments."""
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 20
    frame_size = int(sr * frame_ms / 1000)
    n = len(audio)
    voiced = []
    def is_voiced(frame):
        import struct
        # webrtcvad expects 16-bit PCM mono
        pcm16 = np.clip(frame * 32768, -32768, 32767).astype(np.int16).tobytes()
        return vad.is_speech(pcm16, sample_rate=sr)

    i = 0
    active = False
    seg_start = 0
    while i + frame_size <= n:
        frame = audio[i:i+frame_size]
        v = is_voiced(frame)
        if v and not active:
            active = True
            seg_start = i
        elif not v and active:
            active = False
            seg_end = i
            voiced.append((seg_start, seg_end))
        i += frame_size
    if active:
        voiced.append((seg_start, n))

    # merge/expand segments
    pad = int(sr * pad_ms / 1000)
    merged = []
    for s, e in voiced:
        s2 = max(0, s - pad)
        e2 = min(n, e + pad)
        if merged and s2 <= merged[-1][1] + pad:
            merged[-1] = (merged[-1][0], e2)
        else:
            merged.append((s2, e2))

    # split too-long segments
    final = []
    max_len = int(sr * max_chunk_ms / 1000)
    for s, e in merged:
        length = e - s
        if length <= max_len:
            final.append((s, e))
        else:
            k = s
            while k < e:
                final.append((k, min(k + max_len, e)))
                k += max_len
    return final

def transcribe_chunks(audio: np.ndarray, sr: int, model_size: str, language_hint: Optional[str]) -> Tuple[List[dict], str]:
    m = init_model_if_needed(model_size)
    segments_all = []
    language = language_hint or "auto"
    t0 = time.perf_counter()
    # faster-whisper expects float32 numpy audio
    it = m.model.transcribe(audio, language=language_hint, beam_size=5, vad_filter=False)
    for seg in it[0]:
        segments_all.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    lang = it[1].language or (language_hint if language_hint else "unknown")
    return segments_all, lang
