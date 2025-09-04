from __future__ import annotations
import io
import subprocess
import soundfile as sf
import numpy as np
from typing import Tuple

def read_audio_bytes_to_mono16k(data: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode with ffmpeg to mono PCM float32 at target_sr."""
    # Use ffmpeg in-memory decode
    p = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-ac", "1", "-ar", str(target_sr),
            "-f", "f32le", "pipe:1"
        ],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if p.returncode != 0 or len(p.stdout) == 0:
        raise ValueError(f"ffmpeg decode failed: {p.stderr.decode('utf-8', 'ignore')[:200]}")
    audio = np.frombuffer(p.stdout, dtype=np.float32)
    return audio, target_sr

def write_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()

def ffmpeg_denoise_wav_bytes(wav_bytes: bytes, strength: str = "n") -> bytes:
    """Apply ffmpeg afftdn (noise reduction) + gentle band-pass. strength: 'n','m','a','s'"""
    p = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-af", f"highpass=f=100,lowpass=f=7500,afftdn=nr=12:nf=-25:nt={strength}",
            "-f", "wav", "pipe:1"
        ],
        input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if p.returncode != 0 or len(p.stdout) == 0:
        raise RuntimeError(f"ffmpeg denoise failed: {p.stderr.decode('utf-8', 'ignore')[:200]}")
    return p.stdout
