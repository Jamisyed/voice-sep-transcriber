# Voice Separation + Transcription Microservice

Containerized FastAPI service that accepts an audio file, separates vocals from background (Demucs if available; else ffmpeg noise suppression fallback), and returns a JSON transcription (Whisper via `faster-whisper`).

## Features
- **Endpoint**: `POST /v1/transcribe`
- **Accepts**: multipart/form-data with `file` and optional `config` JSON string
- **Pipeline**: decode → normalize → separation → VAD/chunk → ASR → (diarization optional) → merge → format
- **Separation**: Demucs (if installed) else ffmpeg `afftdn` fallback, clearly flagged
- **ASR**: `faster-whisper` (sizes: tiny/base/small/medium/large-v3); auto language detection with optional `language_hint`
- **VAD**: WebRTC VAD to remove silences and chunk long audio
- **Observability**: request_id, structured logs, per-stage timings
- **Dockerized**: Dockerfile + docker-compose.yml
- **CLI**: `cli/transcribe.py` posts a local file and prints the transcript
- **CPU-only by default**, optional CUDA for `faster-whisper` if available

> **Note on models**: Demucs and diarization models are **optional** and only used if installed (see ADR below). The container runs fine with CPU-only Whisper and ffmpeg fallback.

---

## Quickstart (Docker)

```bash
# 1) Build
docker build -t voice-sep-transcriber .

# 2) Run
docker run --rm -p 8000:8000 -v $(pwd)/sample_audio:/app/sample_audio voice-sep-transcriber

# 3) Transcribe a file
curl -X POST "http://localhost:8000/v1/transcribe"   -F "file=@sample_audio/noisy_sample.wav"   -F 'config={"language_hint":"en","enable_separation":true,"model_size":"small"}'
```

Or use the CLI:

```bash
python cli/transcribe.py sample_audio/noisy_sample.wav --host http://localhost:8000 --model small
```

OpenAPI docs: http://localhost:8000/docs

---

## Local (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional add-ons (GPU, separation, diarization)
# pip install 'faster-whisper[cuda]'  # requires compatible CUDA runtime
# pip install demucs torch torchaudio  # for separation (heavy)
# pip install pyannote.audio           # diarization (heavy, requires HF token for pretrained pipelines)

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## docker-compose (optional)

```bash
docker compose up --build
```

This sets `NUM_WORKERS` and can map a model cache volume.

---

## Request / Response

### Request (multipart)
- `file`: audio binary (wav/mp3/m4a/flac/ogg)
- `config` (optional JSON as **string** field):
```json
{
  "language_hint": "en",
  "enable_separation": true,
  "diarize": false,
  "model_size": "small",
  "target_sr": 16000
}
```

### Response (example)
```json
{
  "request_id": "uuid",
  "duration_sec": 31.2,
  "sample_rate": 16000,
  "pipeline": {
    "separation": {"enabled": true, "method": "demucs", "fallback_used": false},
    "transcription": {"model": "whisper-small"}
  },
  "segments": [{"start": 0.0, "end": 3.1, "text": "hello world"}],
  "text": "hello world",
  "language": "en",
  "timings_ms": {"load": 420, "separation": 1800, "transcription": 4100, "total": 6400}
}
```

### Errors
- 400 invalid file/format
- 413 file too large
- 422 decode failure
- 500 unexpected; includes `request_id` and message

---

## Sample Audio
The repo includes placeholders in `sample_audio/`:
- `noisy_sample.wav` – synthetic tone + noise
- `speech_with_music.wav` – placeholder (please replace with a short clip you own the rights to)
- `office_ambient.wav` – placeholder

> Replace placeholders with 10–30s clips of various noise conditions. The expected behavior: when separation is enabled, vocals should be clearer and ASR WER lower; if Demucs isn't present, ffmpeg denoise runs and `fallback_used=true` is set.

---

## Architecture Decision Record (ADR)

### Why FastAPI + Python?
- Excellent for ML pipelines and async file I/O.
- Built-in OpenAPI docs, easy testing with `httpx`/`pytest`.

### Separation Strategy
- **Primary**: Demucs (SOTA for music/vocals source separation). Heavy dependency (PyTorch), so optional.
- **Fallback**: ffmpeg frequency-domain denoising (`afftdn` + high/low-pass). Not true separation, but robust, fast, and CPU-friendly.

### ASR Model
- **faster-whisper** (CTranslate2) chosen for strong accuracy/speed on CPU, and easy CUDA acceleration.
- Model sizes allow tuning latency vs. accuracy. Default: `small`.

### GPU/CPU Detection
- Detect CUDA via `ctranslate2` device query; otherwise CPU. Mixed-precision if CUDA.
- Demucs used only if its import succeeds and `enable_separation=true`.

### Chunking & VAD
- Resample to 16k mono.
- WebRTC VAD prunes silence, creates chunks (10–30s) with small overlaps; stitched back with timestamps.

### Memory & Concurrency
- Single global model instance per process (lazy-init on first request).
- Use `uvicorn` with N workers (via env `NUM_WORKERS`) for CPU-bound parallelism.
- Large uploads rejected > ~100 MB (configurable).

### Observability
- `loguru` structured logs with `request_id`.
- Timings for: load/decode, separation, transcription, total.

### Failure Modes & Fallbacks
- If Demucs fails or not installed: set `fallback_used=true` and use ffmpeg denoise.
- If ASR fails for a chunk: skip with warning; include partial results.
- If diarization unavailable: flag ignored gracefully.

---

## Testing

```bash
pytest -q
```

---

## Trade-offs
- Demucs/pyannote are big; making them optional keeps the base image lean.
- Using ffmpeg denoise as fallback isn't true separation, but provides a dependable quality/latency baseline on CPU-only hosts.
- Diarization off by default to avoid heavyweight dependencies.

---

## Repo Layout
```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── src/app
│   ├── main.py
│   ├── audio.py
│   ├── separation.py
│   ├── asr.py
│   ├── diarize.py
│   ├── models.py
│   └── version.py
├── cli/transcribe.py
├── tests/test_api.py
└── sample_audio/
    ├── noisy_sample.wav
    ├── speech_with_music.wav  (placeholder)
    └── office_ambient.wav     (placeholder)
```
