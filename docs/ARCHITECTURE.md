# Architecture

## Layers

| Layer | Browser | Server |
| --- | --- | --- |
| Interface | `web/index.html`, `web/src/app.js` (Simple view, Mixer view) | static hosting of `web/` |
| Voice Profile | `web/src/profile.js` (`ProfileStore`) | `voice_engine/profile.py` |
| AI generator | `web/src/ai.js` (Claude via the artifact runtime, or offline rules) | `voice_engine/ai/interpreter.py` (Claude API, or offline rules) |
| Voice analysis | `web/src/analysis.js` | `voice_engine/analysis.py` |
| Audio engine | `web/src/dsp.js` | `voice_engine/dsp/engine.py` |
| Compute backend | JavaScript on the main thread | `backends/cpu.py` (numpy/numba), `backends/rocm.py` (PyTorch on AMD GPU) |
| Transport | — | `server.py`: REST for render/analyze/interpret, WebSocket for live blocks |

Both sides read `shared/voice_profile.schema.json`. The browser gets it through `web/src/schema.generated.js` (`python scripts/gen_schema.py`; CI fails if it is stale).

## One profile, two editors

All edits go through `ProfileStore.set(changes, source)`, where `source` is `ai`, `mixer`, `match`, `load` or `reset`. The store validates and clamps every value against the schema, logs the diff, and notifies listeners: the preview re-renders, the Mixer updates, the live engine reconfigures. The AI is always given the *current* profile and returns only changes, so Mixer edits survive AI turns and AI changes survive Mixer edits.

## Processing chain

```
input gain → noise suppression (STFT, min-tracking noise estimate, 512/256, latency 512)
→ speech detection (pitch periodicity sustained 30 ms + energy over adaptive floor; 30 ms lookahead mute)
→ high-pass → mud cut (350 Hz) → warmth (180 Hz) → presence (4 kHz) → air (10 kHz shelf)
→ pitch shift (two-tap delay line; only when ≠ 0) → de-esser (split band, 4:1 above threshold)
→ compressor (soft knee, feed-forward) + makeup → loudness normalisation (K-weighted, speech-gated, 256-sample frames)
→ output gain → limiter (instant attack; never exceeds the ceiling)
```

Two deviations from the requirement document's suggested order, both deliberate: speech detection runs after noise suppression so it sees the cleaned signal, and output gain sits before the limiter so the output cannot clip.

Every stage keeps its own state and produces identical output for any block size, so the live engine (blocks of 256–2048) and the preview renderer (blocks of 1024) sound the same. `Pipeline.latencySamples` reports the exact delay, which the preview removes to keep Original and Enhanced aligned.

## Backend contract

```python
class ProcessingBackend:
    name: str
    @classmethod
    def available(cls) -> tuple[bool, str]: ...
    def create_pipeline(self, fs: float, params: dict) -> Pipeline: ...

pipeline.configure(params)                 # live profile updates
info = pipeline.process(block, bypass)     # float32 mono block, in place
pipeline.latency_samples
```

The STFT stage is written against a small array-ops interface (`NumpyOps`, `TorchOps`), so the ROCm backend runs the same algorithm on the GPU. New ROCm models plug in by replacing one stage:

| Stage slot | Current implementation | ROCm candidate |
| --- | --- | --- |
| Noise suppression | spectral suppression | learned denoiser (e.g. DeepFilterNet-class model) |
| Speech detection | periodicity + energy | Silero VAD |
| Voice character | EQ + pitch shifter | w-okada/voice-changer model, conservative settings |
| Speech-to-text | — | whisper.cpp with the HIP backend |

Keep `process(block, bypass)` and the `BlockInfo` fields unchanged and the UI, API, tests and metrics continue to work.

## AI layer

The generator never touches audio. It receives the schema with current values, the user's measured voice, the reference target, a deterministic personalised match, and the conversation, and must return `{"reply": ..., "changes": {...}}`. `parse_ai_response` extracts JSON (including fenced blocks), then `validate_changes` drops unknown keys and clamps values. If the model call fails, the server falls back to the offline interpreter and says so in the response `note`.

## Live streaming (server mode)

The browser captures mic blocks with a ScriptProcessor, sends float32 frames over `/ws/stream`, and plays whatever processed block has come back. The reply for block *k* is played in callback *k+1*, so server mode adds one block of latency plus a jitter queue of up to three blocks. Production should move capture to an AudioWorklet and, for desktop use, to native audio I/O with a virtual device.
