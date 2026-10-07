"""HTTP + WebSocket API for the voice engine, and static hosting of the web app.

Endpoints (see docs/API.md):
  GET  /api/health            backend, interpreter, versions
  GET  /api/schema            Voice Profile schema + default reference
  POST /api/profile/validate  clamp/validate a change set
  POST /api/analyze           measure a recording (multipart "audio")
  POST /api/render            enhance a recording with a profile (multipart "audio" + "profile")
  POST /api/interpret         natural language -> profile changes
  GET  /api/benchmark         latency table for the active backend
  WS   /ws/stream             real-time block processing
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .ai import OfflineInterpreter, get_interpreter
from .analysis import analyze_voice
from .audio_io import read_audio, wav_bytes
from .backends import available_backends, get_backend
from .dsp.engine import render_offline
from .metrics import benchmark
from .profile import DEFAULT_REFERENCE, DEFAULTS, GROUPS, SCHEMA, SHARED, full_profile, validate_changes

log = logging.getLogger("voice_engine")
WEB_DIR = Path(os.environ.get("VOICE_WEB_DIR", SHARED.parent / "web"))
MAX_UPLOAD_SECONDS = float(os.environ.get("VOICE_MAX_SECONDS", "120"))
ALLOWED_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)


def _finite(o):
    """JSON cannot carry inf/nan: replace them with None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


class Changes(BaseModel):
    changes: dict = Field(default_factory=dict)


class InterpretBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    profile: dict = Field(default_factory=dict)
    analysis: dict | None = None
    reference: dict | None = None
    history: list = Field(default_factory=list)


def create_app(backend_name: str | None = None, interpreter=None) -> FastAPI:
    backend = get_backend(backend_name)
    interp = interpreter or get_interpreter()
    app = FastAPI(title="Voice Enhancer API", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("VOICE_CORS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
    app.state.backend, app.state.interpreter = backend, interp

    async def load_upload(audio: UploadFile, fs: int | None) -> tuple[np.ndarray, int]:
        raw = await audio.read()
        if not raw:
            raise HTTPException(400, "The audio upload was empty.")
        name = (audio.filename or "").lower()
        if name.endswith(".f32") or audio.content_type == "application/octet-stream" and fs:
            if not fs or fs not in ALLOWED_RATES:
                raise HTTPException(400, f"Raw float32 audio needs a valid fs form field ({', '.join(map(str, ALLOWED_RATES))}).")
            if len(raw) % 4:
                raise HTTPException(400, "Raw float32 audio length is not a multiple of 4 bytes.")
            data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            data = data[: int(fs * MAX_UPLOAD_SECONDS)]
        else:
            try:
                data, fs = read_audio(raw, MAX_UPLOAD_SECONDS)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(415, f"Could not decode the audio file ({e.__class__.__name__}). Use WAV, FLAC or OGG.") from e
        if not np.all(np.isfinite(data)):
            raise HTTPException(400, "The audio contains NaN or infinite samples.")
        return data, int(fs)

    def parse_profile(profile: str | None) -> dict:
        if not profile:
            return dict(DEFAULTS)
        try:
            obj = json.loads(profile)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"profile is not valid JSON: {e.msg}") from e
        return full_profile(obj.get("params", obj) if isinstance(obj, dict) else {})

    @app.get("/api/health")
    def health():
        ok, detail = backend.available()
        return {"ok": True, "version": __version__, "backend": {"name": backend.name, "detail": detail},
                "backends": available_backends(), "interpreter": interp.name, "schemaKeys": len(SCHEMA)}

    @app.get("/api/schema")
    def schema():
        return {"groups": GROUPS, "parameters": SCHEMA, "defaults": DEFAULTS, "reference": DEFAULT_REFERENCE}

    @app.post("/api/profile/validate")
    def validate(body: Changes):
        ok, rejected = validate_changes(body.changes)
        return {"ok": ok, "rejected": rejected}

    @app.post("/api/analyze")
    async def analyze(audio: UploadFile = File(...), fs: int | None = Form(None)):
        data, rate = await load_upload(audio, fs)
        return JSONResponse(_finite({"fs": rate, "analysis": analyze_voice(data, rate)}))

    @app.post("/api/render")
    async def render(audio: UploadFile = File(...), profile: str | None = Form(None), fs: int | None = Form(None),
                     chunk: int = Form(1024), format: str = Form("json")):
        if chunk not in (128, 256, 512, 1024, 2048, 4096):
            raise HTTPException(400, "chunk must be one of 128, 256, 512, 1024, 2048, 4096")
        data, rate = await load_upload(audio, fs)
        params = parse_profile(profile)
        t0 = time.perf_counter()
        out, infos, lat = await asyncio.to_thread(render_offline, data, rate, params, chunk, backend.create_pipeline)
        ms = (time.perf_counter() - t0) * 1000
        if format == "wav":
            return Response(wav_bytes(out, rate), media_type="audio/wav", headers={"X-Processing-Ms": f"{ms:.1f}", "X-Backend": backend.name})
        return JSONResponse(_finite({
            "fs": rate, "chunk": chunk, "latencySamples": lat, "processingMs": ms, "backend": backend.name, "params": params,
            "audio": base64.b64encode(out.astype("<f4").tobytes()).decode(), "infos": [i.to_dict() for i in infos],
        }))

    @app.post("/api/interpret")
    async def interpret(body: InterpretBody):
        ctx = {"profile": full_profile(body.profile), "analysis": body.analysis, "reference": body.reference or DEFAULT_REFERENCE, "history": body.history}
        note, used = None, interp
        try:
            res = await asyncio.to_thread(interp.interpret, body.text, ctx)
        except Exception as e:  # noqa: BLE001 - LLM outage must not break the app
            log.warning("interpreter %s failed: %s", interp.name, e)
            used, note = OfflineInterpreter(), f"{interp.name} failed ({e.__class__.__name__}); the offline interpreter handled this."
            res = used.interpret(body.text, ctx)
        return {"reply": res.get("reply"), "changes": res.get("changes", {}), "rejected": res.get("rejected", []), "interpreter": used.name, "note": note}

    @app.get("/api/benchmark")
    async def bench(chunks: str = "256,512,1024,2048", seconds: float = 6.0):
        try:
            cs = tuple(int(c) for c in chunks.split(","))
        except ValueError as e:
            raise HTTPException(400, "chunks must be comma-separated integers") from e
        rows = await asyncio.to_thread(benchmark, backend, DEFAULTS, 48000, cs, min(seconds, 30.0), 1)
        return {"backend": backend.name, "rows": rows}

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket):
        """Protocol: client sends {"type":"start","sampleRate":N,"profile":{...}} then binary float32 blocks.
        Each binary block is answered with the processed block (same length). Text messages
        {"type":"profile","profile":{...}} and {"type":"bypass","value":bool} update the stream.
        Every 4th block the server also sends {"type":"info", ...meters}."""
        await ws.accept()
        pl, bypass, n = None, False, 0
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    try:
                        m = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        await ws.send_text(json.dumps({"type": "error", "message": "messages must be JSON"}))
                        continue
                    if m.get("type") == "start":
                        rate = int(m.get("sampleRate", 48000))
                        if rate not in ALLOWED_RATES:
                            await ws.send_text(json.dumps({"type": "error", "message": f"unsupported sampleRate {rate}"}))
                            continue
                        pl = backend.create_pipeline(rate, full_profile(m.get("profile")))
                        bypass = bool(m.get("bypass", False))
                        await ws.send_text(json.dumps({"type": "ready", "backend": backend.name, "latencySamples": pl.latency_samples}))
                    elif m.get("type") == "profile" and pl:
                        pl.configure(full_profile(m.get("profile")))
                    elif m.get("type") == "bypass":
                        bypass = bool(m.get("value"))
                    continue
                data = msg.get("bytes")
                if data is None:
                    continue
                if pl is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "send a start message first"}))
                    continue
                if len(data) % 4 or len(data) > 4 * 16384:
                    await ws.send_text(json.dumps({"type": "error", "message": "blocks must be float32, at most 16384 samples"}))
                    continue
                b = np.frombuffer(data, dtype="<f4").copy()
                b[~np.isfinite(b)] = 0
                t0 = time.perf_counter()
                info = pl.process(b, bypass)
                ms = (time.perf_counter() - t0) * 1000
                await ws.send_bytes(b.astype("<f4").tobytes())
                n += 1
                if n % 4 == 0:
                    await ws.send_text(json.dumps(_finite({"type": "info", "processMs": ms, **info.to_dict()})))
        except WebSocketDisconnect:
            pass

    if WEB_DIR.exists():
        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

        app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
    return app


app = None  # created lazily by `voice-engine serve` / uvicorn factory


def factory():
    return create_app()
