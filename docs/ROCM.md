# Running on AMD ROCm

## 1. Check the machine

```bash
rocminfo | grep -E "Marketing Name|gfx"
rocm-smi
```

Record the GPU, `gfx` target and ROCm version. These go into the report (proposal section 6, "Environment setup").

## 2. Install PyTorch for ROCm

Use the wheel index that matches the installed ROCm version, as listed on pytorch.org ("Get started", ROCm column), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2   # match your ROCm version
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

ROCm PyTorch exposes AMD GPUs through the `torch.cuda` API; `torch.version.hip` must not be `None`.

Alternatively use the container: `docker build -f docker/Dockerfile.rocm --build-arg ROCM_PYTORCH_TAG=<tag> -t voice-engine:rocm .` and run with `--device=/dev/kfd --device=/dev/dri --group-add video`.

## 3. Verify

```bash
pip install -e server
python scripts/verify_rocm.py
```

The script writes `rocm_environment.json` (versions, GPU) and `rocm_benchmark.csv` (p50/p95 latency, real-time factor, overruns per chunk size for CPU and ROCm), and checks that the ROCm backend's output matches the CPU backend within 1e-4. Keep both files for the performance report.

## 4. Serve

```bash
VOICE_BACKEND=rocm voice-engine serve --host 0.0.0.0
```

`/api/health` shows the backend in use. In the web app, Mixer → Engine shows "Server (rocm)".

## What runs on the GPU today, and what should come next

Today the STFT noise-suppression stage runs through PyTorch on the device; the per-sample stages (filters, compressor, limiter) stay on the CPU because they are recursive and cheap. A single 512-point FFT per 5 ms does not need a GPU, so expect the ROCm backend to be *slower* than the CPU backend for one stream until heavier models are added. Report that honestly: the proposal commits to measured results, not assumed speed-ups.

The GPU earns its place with the models listed in docs/ARCHITECTURE.md (learned denoiser, Silero VAD, w-okada voice changer, whisper.cpp HIP) and with batch rendering of many files. Each replaces one stage behind the same `ProcessingBackend` contract.
