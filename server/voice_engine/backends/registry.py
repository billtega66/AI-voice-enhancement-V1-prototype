from __future__ import annotations

import os

from .cpu import CpuBackend
from .rocm import RocmBackend

BACKENDS = {"cpu": CpuBackend, "rocm": RocmBackend}


def available_backends() -> list[dict]:
    return [cls().describe() for cls in BACKENDS.values()]


def get_backend(name: str | None = None):
    """name: 'cpu', 'rocm' or 'auto' (default; env VOICE_BACKEND). 'auto' picks ROCm when usable."""
    name = (name or os.environ.get("VOICE_BACKEND", "auto")).lower()
    if name == "auto":
        return RocmBackend() if RocmBackend.available()[0] else CpuBackend()
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)} or auto")
    cls = BACKENDS[name]
    ok, why = cls.available()
    if not ok:
        raise RuntimeError(f"backend {name!r} unavailable: {why}")
    return cls()
