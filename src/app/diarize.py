from __future__ import annotations
from typing import List, Dict

def maybe_diarize(segments: List[Dict], enable: bool = False) -> List[Dict]:
    """
    Placeholder for diarization with pyannote.audio if installed.
    Currently returns segments unchanged with speaker='SPEAKER_00'.
    """
    if not enable:
        return segments
    # TODO: integrate pyannote pipeline; assign speaker labels per segment
    for s in segments:
        s["speaker"] = "SPEAKER_00"
    return segments
