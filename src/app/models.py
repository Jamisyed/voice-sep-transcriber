from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class SeparationInfo(BaseModel):
    enabled: bool
    method: str
    fallback_used: bool = False

class TranscriptionInfo(BaseModel):
    model: str

class PipelineInfo(BaseModel):
    separation: SeparationInfo
    transcription: TranscriptionInfo

class Segment(BaseModel):
    start: float
    end: float
    text: str

class Timings(BaseModel):
    load: int
    separation: int
    transcription: int
    total: int

class TranscribeResponse(BaseModel):
    request_id: str
    duration_sec: float
    sample_rate: int
    pipeline: PipelineInfo
    segments: List[Segment]
    text: str
    language: str
    timings_ms: Timings

class ConfigInput(BaseModel):
    language_hint: Optional[str] = None
    enable_separation: bool = True
    diarize: bool = False
    model_size: str = "small"
    target_sr: int = 16000
