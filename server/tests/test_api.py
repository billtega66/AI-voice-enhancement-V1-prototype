"""HTTP + WebSocket API tests (FastAPI TestClient, no network)."""
import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_engine.ai import OfflineInterpreter
from voice_engine.audio_io import wav_bytes
from voice_engine.dsp import render_offline
from voice_engine.profile import DEFAULTS, SCHEMA
from voice_engine.server import create_app


class FailingInterpreter:
    name = "claude"

    def interpret(self, text, ctx):
        raise TimeoutError("upstream timeout")


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app("cpu", OfflineInterpreter()))


def f32(x):
    return ("take.f32", np.asarray(x, "<f4").tobytes(), "application/octet-stream")


def test_health_and_schema(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["backend"]["name"] == "cpu" and h["interpreter"] == "offline" and h["schemaKeys"] == len(SCHEMA)
    assert {b["name"] for b in h["backends"]} == {"cpu", "rocm"}
    s = client.get("/api/schema").json()
    assert s["defaults"]["targetLufs"] == -16 and "integratedLufs" in s["reference"]


def test_validate(client):
    r = client.post("/api/profile/validate", json={"changes": {"warmthDb": 99, "nope": 1}}).json()
    assert r == {"ok": {"warmthDb": 9}, "rejected": ["nope"]}


def test_render_raw_float32_matches_engine(client, sample):
    x = sample.data[:48000 * 4]
    prof = {**DEFAULTS, "warmthDb": 3}
    r = client.post("/api/render", files={"audio": f32(x)}, data={"fs": "48000", "profile": json.dumps(prof), "chunk": "1024"})
    assert r.status_code == 200, r.text
    j = r.json()
    out = np.frombuffer(base64.b64decode(j["audio"]), "<f4")
    ref, infos, lat = render_offline(x, 48000, prof)
    assert np.array_equal(out, ref) and j["latencySamples"] == lat and len(j["infos"]) == len(infos)
    assert set(j["infos"][0]) >= {"speech", "inDb", "outLufs", "compDb", "deEssDb", "limDb", "loudDb", "nsDb"}


def test_render_wav_upload_and_wav_response(client, sample):
    wav = wav_bytes(sample.data[:48000 * 2], 48000)
    r = client.post("/api/render", files={"audio": ("take.wav", wav, "audio/wav")}, data={"format": "wav"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"


@pytest.mark.parametrize("files,data,code,msg", [
    ({"audio": ("take.f32", b"", "application/octet-stream")}, {"fs": "48000"}, 400, "empty"),
    ({"audio": ("take.f32", b"\x00\x00\x00", "application/octet-stream")}, {"fs": "48000"}, 400, "multiple of 4"),
    ({"audio": ("take.f32", np.zeros(10, "<f4").tobytes(), "application/octet-stream")}, {"fs": "12345"}, 400, "valid fs"),
    ({"audio": ("take.f32", np.array([np.nan] * 8, "<f4").tobytes(), "application/octet-stream")}, {"fs": "48000"}, 400, "NaN"),
    ({"audio": ("take.wav", b"not audio", "audio/wav")}, {}, 415, "Could not decode"),
    ({"audio": ("take.f32", np.zeros(4800, "<f4").tobytes(), "application/octet-stream")}, {"fs": "48000", "profile": "{bad"}, 400, "not valid JSON"),
    ({"audio": ("take.f32", np.zeros(4800, "<f4").tobytes(), "application/octet-stream")}, {"fs": "48000", "chunk": "1000"}, 400, "chunk must be"),
])
def test_render_rejects_bad_input(client, files, data, code, msg):
    r = client.post("/api/render", files=files, data=data)
    assert r.status_code == code and msg in r.json()["detail"]


def test_analyze(client, sample):
    a = client.post("/api/analyze", files={"audio": f32(sample.data)}, data={"fs": "48000"}).json()["analysis"]
    assert a["ok"] and 110 < a["f0MedianHz"] < 175 and a["formants"] is None


def test_interpret_offline_and_keeps_profile(client):
    prof = {**DEFAULTS, "warmthDb": 4.5, "compRatio": 4}
    r = client.post("/api/interpret", json={"text": "Keep everything else but make it slightly clearer.", "profile": prof}).json()
    assert r["interpreter"] == "offline" and set(r["changes"]) <= {"presenceDb", "mudDb"} and r["changes"]["presenceDb"] == 1


def test_interpret_falls_back_when_llm_fails():
    c = TestClient(create_app("cpu", FailingInterpreter()))
    r = c.post("/api/interpret", json={"text": "make it warmer"}).json()
    assert r["interpreter"] == "offline" and "TimeoutError" in r["note"] and r["changes"]["warmthDb"] == 2


def test_interpret_validates_input(client):
    assert client.post("/api/interpret", json={"text": ""}).status_code == 422


def test_benchmark_endpoint(client):
    rows = client.get("/api/benchmark", params={"chunks": "512,1024", "seconds": 2}).json()["rows"]
    assert [r["chunk"] for r in rows] == [512, 1024] and all(r["rtf_p95"] < 1 for r in rows)


def test_websocket_stream_matches_offline_engine(client, sample):
    x = sample.data[:48000 * 2]
    prof = {**DEFAULTS, "presenceDb": 2}
    out, infos = [], []
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"type": "start", "sampleRate": 48000, "profile": prof}))
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready" and ready["backend"] == "cpu"
        for i in range(0, len(x) - 1024 + 1, 1024):
            ws.send_bytes(x[i:i + 1024].astype("<f4").tobytes())
            out.append(np.frombuffer(ws.receive_bytes(), "<f4"))
            if (i // 1024 + 1) % 4 == 0:
                infos.append(json.loads(ws.receive_text()))
    got = np.concatenate(out)
    ref, _, lat = render_offline(x, 48000, prof)
    # live output is the raw stream (not latency-compensated): compare after shifting
    assert np.max(np.abs(got[lat:] - ref[:len(got) - lat])) < 1e-6
    assert infos and infos[0]["type"] == "info" and "processMs" in infos[0]


def test_websocket_bypass_and_profile_update(client, sample):
    x = sample.data[48000:48000 + 4096]
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"type": "start", "sampleRate": 48000, "profile": {}, "bypass": True}))
        ws.receive_text()
        ws.send_bytes(x[:1024].astype("<f4").tobytes())
        assert np.array_equal(np.frombuffer(ws.receive_bytes(), "<f4"), x[:1024])
        ws.send_text(json.dumps({"type": "bypass", "value": False}))
        ws.send_text(json.dumps({"type": "profile", "profile": {"outputGainDb": -24, "limiterEnabled": False}}))
        ws.send_bytes(x[1024:2048].astype("<f4").tobytes())
        assert not np.array_equal(np.frombuffer(ws.receive_bytes(), "<f4"), x[1024:2048])


def test_websocket_errors(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(np.zeros(16, "<f4").tobytes())
        assert "start" in json.loads(ws.receive_text())["message"]
        ws.send_text("not json")
        assert "JSON" in json.loads(ws.receive_text())["message"]
        ws.send_text(json.dumps({"type": "start", "sampleRate": 12345}))
        assert "sampleRate" in json.loads(ws.receive_text())["message"]
        ws.send_text(json.dumps({"type": "start", "sampleRate": 48000}))
        ws.receive_text()
        ws.send_bytes(b"\x00\x01\x02")
        assert "float32" in json.loads(ws.receive_text())["message"]


def test_serves_web_app(client):
    r = client.get("/")
    assert r.status_code == 200 and "Voice Enhancer" in r.text and "src/server-client.js" in r.text
    assert client.get("/src/dsp.js").status_code == 200
