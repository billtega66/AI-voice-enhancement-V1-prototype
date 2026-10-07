"""AMD ROCm backend.

PyTorch's ROCm build exposes AMD GPUs through the ``torch.cuda`` API, and
``torch.version.hip`` is set. This backend runs the STFT noise-suppression stage on
the GPU through the same algorithm as the CPU backend (see ``NoiseSuppressor``'s ops
interface), and is the place to add heavier ROCm models: a learned denoiser, Silero
VAD, or the w-okada voice-changer model. Each of those should replace one stage slot
and keep ``Pipeline.process`` unchanged.

Not exercised in CI (no AMD GPU there). Run ``python scripts/verify_rocm.py`` on the
AMD machine: it checks the device, then checks this backend against the CPU backend
and reports latency.
"""
from __future__ import annotations

import numpy as np

from ..dsp.engine import Pipeline
from .base import ProcessingBackend


class TorchOps:
    """Array ops for NoiseSuppressor backed by torch tensors on ``device``."""

    name = "torch"

    def __init__(self, device: str = "cuda"):
        import torch

        self.torch, self.device = torch, torch.device(device)

    def from_host(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float64), device=self.device)

    def to_host(self, a):
        return a.detach().to("cpu").numpy().astype(np.float64)

    def zeros(self, n):
        return self.torch.zeros(n, dtype=self.torch.float64, device=self.device)

    def ones(self, n):
        return self.torch.ones(n, dtype=self.torch.float64, device=self.device)

    def fft(self, a):
        return self.torch.fft.fft(a)

    def ifft(self, a):
        return self.torch.fft.ifft(a)

    def real(self, a):
        return a.real

    def imag(self, a):
        return a.imag

    def minimum(self, a, b):
        return self.torch.minimum(a, b)

    def where(self, c, a, b):
        return self.torch.where(c, a, b)

    def clamp_min(self, a, v):
        return self.torch.clamp_min(a, v)

    def flip(self, a):
        return self.torch.flip(a, dims=[0])

    def concat(self, a, b):
        return self.torch.cat([a, b])


class RocmBackend(ProcessingBackend):
    name = "rocm"

    def __init__(self, device: str = "cuda:0"):
        self.device = device

    @classmethod
    def available(cls):
        try:
            import torch
        except ImportError:
            return False, "PyTorch is not installed (install the ROCm build: see docs/ROCM.md)"
        hip = getattr(torch.version, "hip", None)
        if not hip:
            return False, f"PyTorch {torch.__version__} is not a ROCm build"
        if not torch.cuda.is_available():
            return False, f"ROCm {hip} build found, but no AMD GPU is visible"
        return True, f"ROCm {hip}, {torch.cuda.get_device_name(0)}"

    def create_pipeline(self, fs, params):
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"ROCm backend unavailable: {why}")
        pl = Pipeline(fs, params, ops=TorchOps(self.device))
        pl.backend_name = self.name
        return pl
