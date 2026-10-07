"""Latency / throughput measurement used by the CLI benchmark, the API and the tests."""
from __future__ import annotations

import time

import numpy as np

from .dsp.samples import make_sample_voice


def percentiles(xs, qs=(50, 95)):
    a = np.asarray(xs, dtype=np.float64)
    return {f"p{q}": float(np.percentile(a, q)) if a.size else float("nan") for q in qs}


def benchmark(backend, params: dict, fs: int = 48000, chunks=(256, 512, 1024, 2048), seconds: float = 14.0, passes: int = 2):
    """Process the sample voice at each chunk size; report per-chunk cost, real-time factor and overruns."""
    sample = make_sample_voice(fs, seconds)
    rows = []
    warm = backend.create_pipeline(fs, params)  # JIT warm-up, excluded from timing
    warm.process(np.zeros(1024, np.float32))
    for n in chunks:
        pl = backend.create_pipeline(fs, params)
        budget_ms = n / fs * 1000
        times, over = [], 0
        t_all = time.perf_counter()
        for _ in range(passes):
            for i in range(0, len(sample.data) - n + 1, n):
                b = sample.data[i:i + n].copy()
                t0 = time.perf_counter()
                pl.process(b)
                dt = (time.perf_counter() - t0) * 1000
                times.append(dt)
                over += dt > budget_ms
        total_ms = (time.perf_counter() - t_all) * 1000
        pc = percentiles(times)
        rows.append({
            "backend": backend.name, "chunk": n, "budget_ms": budget_ms, "mean_ms": total_ms / len(times),
            "p50_ms": pc["p50"], "p95_ms": pc["p95"], "rtf_p95": pc["p95"] / budget_ms, "overruns": int(over),
            "chunks": len(times), "pipeline_latency_ms": pl.latency_samples / fs * 1000,
            "est_end_to_end_ms": 2 * budget_ms + pl.latency_samples / fs * 1000 + pc["p50"],
        })
    return rows
