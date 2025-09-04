import io, json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_transcribe_empty():
    r = client.post('/v1/transcribe', files={'file': ('empty.wav', b'')})
    assert r.status_code == 400

def test_transcribe_fake_wav_header():
    fake = b'RIFFxxxxWAVEfmt ' + b'\x00'*100
    r = client.post('/v1/transcribe', files={'file': ('bad.wav', fake)})
    assert r.status_code in (422, 500)
