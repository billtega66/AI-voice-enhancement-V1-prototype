# AI gateway

The AI layer follows the pattern of the reference project (gjn12-31/aiid), reimplemented in Python for this server: one OpenAI-compatible `/chat/completions` endpoint, configured only through environment variables, called only from the server.

```
browser ──POST /api/interpret──► voice-engine ──POST {LLM_BASE_URL}/chat/completions──► Bailian DeepSeek  |  vLLM on ROCm  |  other
                                     │  1. reserve budget (daily + per browser)
                                     │  2. system = fixed rules + schema + current profile; user = JSON data (untrusted)
                                     │  3. JSON mode, temperature 0, reasoning off, timeout, finish_reason == "stop"
                                     │  4. validate structure (pydantic) then meaning (known keys, ranges, pitch guard)
                                     └─ any failure ──► offline interpreter + a visible note; settings stay valid
```

## OpenAI API (local Mac / CPU setup)

Create an API key at https://platform.openai.com/api-keys. Activate the project virtual environment with `source .venv/bin/activate`, copy `.env.example` to `.env` if you do not already have one, and use:

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=your-api-key-here
LLM_THINKING_PARAM=none
VOICE_BACKEND=cpu
```

`gpt-4.1-mini` supports the gateway's Chat Completions and JSON mode request format. Set `LLM_THINKING_PARAM=none` so provider-specific reasoning fields are omitted. This app reads `LLM_API_KEY`, rather than `OPENAI_API_KEY`. Keep your real key in the ignored `.env` file on the server.

Run `make serve`, then open http://127.0.0.1:8000 in Chrome or Edge. Restart after editing `.env`. With a blank key, the app runs with the offline interpreter; with a key, `/api/health` should report `interpreter: "gateway"` and `model: "gpt-4.1-mini"`. The API converts text requests into voice settings; audio enhancement runs locally on the CPU or in the browser.

On macOS, use Python 3.12 for a predictable scientific Python environment. SoundFile's macOS wheels normally bundle libsndfile, so the Linux `apt` command in the quick start is unnecessary. ROCm requires supported AMD hardware and is not the Mac backend.

## Option A: DeepSeek on Alibaba Bailian (same model as the reference project)

```bash
cp .env.example .env
# LLM_BASE_URL=<your Bailian / gateway HTTPS URL ending in /v1>
# LLM_MODEL=bailian/deepseek-v4.1-flash
# LLM_API_KEY=<key>
# LLM_THINKING_PARAM=enable_thinking
make serve
python scripts/check_llm.py   # 6 paid calls
```

The model identifier must match what your gateway exposes; `bailian/deepseek-v4.1-flash` is the identifier used by the reference project's gateway. Use an HTTPS gateway: over plain HTTP the bearer key and the user's requests travel unencrypted.

## Option B: local model with vLLM on the AMD GPU

This keeps the whole system on AMD hardware: the voice engine's ROCm backend and the language model share the GPU, and no request leaves the machine.

```bash
docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8g \
  -p 127.0.0.1:8001:8001 -v ~/.cache/huggingface:/root/.cache/huggingface \
  rocm/vllm:latest vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.6
```

```bash
# .env
LLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=Qwen/Qwen3-8B
LLM_API_KEY=EMPTY
LLM_THINKING_PARAM=chat_template_kwargs
```

Or start both services together: `docker compose -f docker/compose.rocm.yaml up -d --build`.

Notes:

- `LLM_THINKING_PARAM=chat_template_kwargs` sends `{"chat_template_kwargs": {"enable_thinking": false}}`, which Qwen3-style chat templates use to skip hidden reasoning. For models without that switch, set `none`.
- Choose the model by GPU memory. An 8B model in 16-bit needs roughly 16 GB plus cache. Lower `--gpu-memory-utilization` if the voice engine shares the GPU. Record the model, image tag and memory use in the report.
- Check that the model's licence allows your use.
- This path has been tested against a stand-in OpenAI-compatible server in CI, not against vLLM on real AMD hardware. Run `python scripts/check_llm.py` on the GPU machine.

## Limits

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_DAILY_CALL_LIMIT` | 200 | attempts per UTC day, all users; failed attempts count |
| `LLM_PER_CLIENT_PER_MIN` | 8 | attempts per browser (cookie `voice_guest`) per minute |
| `LLM_TIMEOUT_S` | 35 | per-request deadline |
| `VOICE_DB_PATH` | `./data/voice.sqlite` | ledger of attempts (persists across restarts) |

When a limit is hit, requests fall back to the offline interpreter with an explanation. Visitors can clear cookies, so add IP or edge limits before public deployment.

## Selection order

1. `LLM_BASE_URL` + `LLM_MODEL` + `LLM_API_KEY` set → gateway
2. otherwise `ANTHROPIC_API_KEY` set → Claude
3. otherwise → offline keyword interpreter

`/api/health` reports `interpreter` and `model`; the chat header in the app shows them.
