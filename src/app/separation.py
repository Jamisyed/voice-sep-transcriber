from __future__ import annotations
import importlib
from .audio import write_wav_bytes, ffmpeg_denoise_wav_bytes

def separate_vocals_if_available(wav_bytes: bytes):
    """Try Demucs for vocals; else ffmpeg denoise fallback. Returns (wav_bytes, method, fallback_used)."""
    # Try demucs dynamically
    demucs_mod = importlib.util.find_spec("demucs")
    if demucs_mod is not None:
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            import torch
            import torchaudio
            import io

            model = get_model(name="htdemucs")
            # load wav into tensor
            wav, sr = torchaudio.load(io.BytesIO(wav_bytes))
            wav = wav.mean(dim=0, keepdim=True)  # mono
            with torch.no_grad():
                sources = apply_model(model, wav[None], device="cuda" if torch.cuda.is_available() else "cpu", progress=False)[0]
            # demucs order typically: drums, bass, other, vocals (depends on model); get vocals by name if available
            if hasattr(model, 'sources') and 'vocals' in model.sources:
                vocals_idx = model.sources.index('vocals')
            else:
                vocals_idx = -1  # heuristic
            vocals = sources[:, vocals_idx] if vocals_idx >= 0 else sources.mean(dim=1)
            import numpy as np
            vocals_np = vocals.squeeze().cpu().numpy()
            out_bytes = write_wav_bytes(vocals_np, sr)
            return out_bytes, "demucs", False
        except Exception:
            pass
    # Fallback
    out_bytes = ffmpeg_denoise_wav_bytes(wav_bytes, strength="m")
    return out_bytes, "ffmpeg-denoise", True
