from __future__ import annotations

import platform

from ..dsp.engine import Pipeline
from .base import ProcessingBackend


class CpuBackend(ProcessingBackend):
    """numpy + numba on the CPU. Sample-accurate match with the browser engine (web/src/dsp.js)."""

    name = "cpu"

    @classmethod
    def available(cls):
        return True, f"numpy/numba on {platform.processor() or platform.machine()}"

    def create_pipeline(self, fs, params):
        return Pipeline(fs, params)
