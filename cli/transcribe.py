#!/usr/bin/env python3
import argparse
import json
import requests
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Transcribe an audio file via the API")
    ap.add_argument("path", help="Path to audio file")
    ap.add_argument("--host", default="http://localhost:8000", help="API host")
    ap.add_argument("--model", default="small", choices=["tiny","base","small","medium","large-v3"], help="Whisper model size")
    ap.add_argument("--language", default=None, help="Language hint, e.g., en, ur, hi")
    ap.add_argument("--no-sep", action="store_true", help="Disable separation")
    args = ap.parse_args()

    p = Path(args.path).expanduser()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    url = f"{args.host}/v1/transcribe"
    cfg = {
        "language_hint": args.language,
        "enable_separation": not args.no_sep,
        "diarize": False,
        "model_size": args.model,
        "target_sr": 16000
    }
    files = {
        "file": (p.name, p.read_bytes(), "application/octet-stream"),
        "config": (None, json.dumps(cfg)),
    }
    r = requests.post(url, files=files, timeout=600)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
