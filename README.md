# AI Voice Enhancement — V1 Prototype

Real-time voice enhancement that keeps the speaker's identity. A user describes how they want to sound in plain language; an AI turns the request into a structured **Voice Profile**; one DSP engine applies it to the user's own voice, live or on a recording. Expert users open the **Mixer** and edit the same profile directly.

Built for the AMD ROCm Student Project (Direction 2: Multimodal Content Creation).

```
User prompt ──► AI Voice Profile Generator ──► Voice Profile (single source of truth) ◄── Mixer
                 (Claude or offline rules)            │
                                                      ▼
                     Browser engine (dsp.js)   or   voice_engine server ──► CPU backend (numpy/numba)
                                                                       └──► ROCm backend (PyTorch on AMD GPU)
```

## What is in this repository

| Path | Contents |
| --- | --- |
| `shared/` | `voice_profile.schema.json` (parameters, ranges, defaults, and creative module descriptors) and `reference_profile.json` (podcast target). Both the browser and the server load these files. |
| `web/` | The web app: AI chat, preview with Original/Enhanced comparison, Use voice, live microphone, Mixer, voice analysis, self-test. Runs fully in the browser, or uses the server when served by it. |
| `server/voice_engine/` | Python engine and API: sample-accurate port of the browser DSP, voice analysis, AI interpreter, CPU and ROCm backends, FastAPI REST + WebSocket server, CLI. |
| `tests/` | Node runner for the browser self-test, browser/server parity fixtures, Playwright end-to-end tests. Server unit and API tests are in `server/tests/`. |
| `scripts/` | Schema generator, single-file web build, `verify_rocm.py` for the AMD machine. |
| `docker/` | CPU and ROCm images. |
| `docs/` | Architecture, API, ROCm deployment, testing and evaluation notes. |

## Quick start

Requirements: Python 3.10+, Node.js 18+ (for tests only), `libsndfile` (`apt install libsndfile1`).

```bash
git clone https://github.com/billtega66/AI-voice-enhancement-V1-prototype
cd AI-voice-enhancement-V1-prototype
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e "server[dev,llm]" + Playwright Chromium
make serve            # http://127.0.0.1:8000
```

Open the address in Chrome or Edge and use headphones for live monitoring. Without the server, `make web` produces `dist/voice-enhancer.html`, which runs on its own from any static host or from disk.

### AI gateway (OpenAI, DeepSeek on Bailian, or vLLM on the AMD GPU)

The AI runs through any OpenAI-compatible `/chat/completions` endpoint, configured in `.env` (see [docs/LLM_GATEWAY.md](docs/LLM_GATEWAY.md)):

```bash
cp .env.example .env     # set LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
make serve               # reads .env automatically; the key never reaches the browser
python scripts/check_llm.py   # optional live check (6 paid calls)
```

- For OpenAI on a local Mac, see [the OpenAI setup](docs/LLM_GATEWAY.md#openai-api-local-mac--cpu-setup): use `https://api.openai.com/v1`, `gpt-4.1-mini`, and `LLM_THINKING_PARAM=none`.
- `LLM_MODEL=bailian/deepseek-v4.1-flash` uses the same model as the reference project.
- Pointing `LLM_BASE_URL` at `vllm serve` on ROCm keeps the language model on the AMD GPU.
- Without a gateway, the server uses Claude if `ANTHROPIC_API_KEY` is set, and otherwise the offline keyword interpreter. The UI shows which one is active.

Every AI answer passes structural and semantic validation before it reaches the engine. A rejected answer, a timeout or an exhausted budget falls back to the offline interpreter, with a note in the chat.

### AMD ROCm

See [docs/ROCM.md](docs/ROCM.md). In short: install the ROCm build of PyTorch, then

```bash
python scripts/verify_rocm.py          # device check, CPU-vs-GPU parity, latency CSV
VOICE_BACKEND=rocm make serve
```

## Command line

```bash
cd server
voice-engine enhance input.wav output.wav --prompt "warmer, podcast, less keyboard noise"
voice-engine analyze input.wav
voice-engine benchmark --chunks 256,512,1024,2048 --csv ../benchmark.csv
voice-engine interpret "slightly deeper and reduce the sharp S sounds"
voice-engine backends
```

## Testing

```bash
make test        # all of the below
make test-js     # browser engine self-test in Node (21 checks)
make test-py     # server stages, API, WebSocket, CLI, AI gateway, and browser/server parity (84 tests)
make test-e2e    # Chromium with a fake microphone: standalone, served, and served with an LLM gateway (5 scenarios)
```

The parity tests render the same input through `web/src/dsp.js` and `server/voice_engine` and require the outputs to match within 1e-5, so a profile sounds the same whichever engine runs it. See [docs/TESTING.md](docs/TESTING.md) for what each suite covers and how it maps to the functional requirements.

## Status and limits

Working: Mixer controls are connected to processing; AI suggestions are previewed as Suggested/Gentler candidates before applying to the same profile. Creative mode adds metallic modulation, distortion, and echo with selectable order. Recording, playback, A/B, Use voice, live microphone, bypass, error messages, and CPU server REST/WebSocket streaming are implemented. See `docs/LLM_GATEWAY.md` for the capability catalog and preview workflow.

Not yet implemented:

- Virtual microphone output. This needs a desktop build, because a web page cannot create an audio device.
- Formant measurement.
- Learned models: Silero VAD, a neural denoiser and the w-okada voice changer. Their stage slots exist in the pipeline (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

The ROCm backend has **not been run on AMD hardware** by this commit. Its code path is contract-tested against the CPU backend with a stand-in tensor library. `scripts/verify_rocm.py` is the first thing to run on the GPU machine.

The reference profile in `shared/reference_profile.json` contains built-in podcast defaults. The requirement document left the measured reference values as a placeholder; replace them with the team's measurements.

## Team

Ben (Nguyen Dinh Bac), Bill (Do Kien Quoc), Juvia (Le Thi Mai Trang). Licence to be confirmed after the dependency and model licence review described in the proposal.
