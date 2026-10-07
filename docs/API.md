# API

Base URL: the address printed by `voice-engine serve` (default `http://127.0.0.1:8000`). Profiles are objects whose keys are listed in `shared/voice_profile.schema.json`; missing keys take defaults, invalid ones are dropped.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/health` | — | `{ok, version, backend:{name, detail}, backends[], interpreter, schemaKeys}` |
| GET | `/api/schema` | — | `{groups, parameters, defaults, reference}` |
| POST | `/api/profile/validate` | `{"changes": {...}}` | `{ok:{...clamped}, rejected:[keys]}` |
| POST | `/api/analyze` | multipart `audio` (WAV/FLAC/OGG, or raw `.f32` with `fs`) | `{fs, analysis}` |
| POST | `/api/render` | multipart `audio`, `profile` (JSON), `fs` (raw only), `chunk` (128–4096), `format` (`json`/`wav`) | JSON `{fs, latencySamples, processingMs, backend, params, audio (base64 float32), infos[]}` or `audio/wav` |
| POST | `/api/interpret` | `{text, profile, analysis?, reference?, history?}` | `{reply, changes, rejected, interpreter, note}` |
| GET | `/api/benchmark` | `?chunks=256,512,1024&seconds=6` | `{backend, rows[]}` |
| WS | `/ws/stream` | see below | processed blocks + info messages |

Errors use HTTP status codes with a readable `detail`: 400 for bad input (empty upload, wrong length, unsupported sample rate, NaN samples, invalid profile JSON, bad chunk), 415 for undecodable audio, 422 for schema violations in JSON bodies.

## WebSocket `/ws/stream`

1. Client sends text `{"type":"start","sampleRate":48000,"profile":{...},"bypass":false}`.
2. Server replies `{"type":"ready","backend":"cpu","latencySamples":1952}`.
3. Client sends binary frames: little-endian float32 mono, up to 16384 samples.
4. Server answers each frame with a processed frame of the same length, in order. Every fourth frame it also sends `{"type":"info", processMs, speech, inDb, outDb, outLufs, nsDb, deEssDb, compDb, limDb, loudDb, ...}`.
5. At any time: `{"type":"profile","profile":{...}}` or `{"type":"bypass","value":true}`.

Bypass returns the input unchanged. Errors arrive as `{"type":"error","message":...}` and do not close the socket.
