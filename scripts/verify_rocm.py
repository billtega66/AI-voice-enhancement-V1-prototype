#!/usr/bin/env python3
"""Run this on the AMD GPU machine after installing ROCm PyTorch (docs/ROCM.md).

It records the evidence the proposal's section 6 asks for:
  1. ROCm/PyTorch versions and the GPU the backend will use
  2. that the ROCm backend produces the same audio as the CPU backend
  3. latency per chunk size on both backends (CSV for the performance report)
Exit code 0 means every check passed.
"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import numpy as np  # noqa: E402

from voice_engine.backends import CpuBackend, RocmBackend  # noqa: E402
from voice_engine.dsp import make_sample_voice, render_offline  # noqa: E402
from voice_engine.metrics import benchmark  # noqa: E402
from voice_engine.profile import DEFAULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="rocm_benchmark.csv")
    ap.add_argument("--json", default="rocm_environment.json")
    args = ap.parse_args()

    ok, detail = RocmBackend.available()
    env = {"python": platform.python_version(), "machine": platform.machine(), "rocm_backend": detail}
    try:
        import torch
        env.update(torch=torch.__version__, hip=getattr(torch.version, "hip", None),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        env["torch"] = None
    print(json.dumps(env, indent=2))
    Path(args.json).write_text(json.dumps(env, indent=2))
    if not ok:
        print(f"\nFAIL: ROCm backend unavailable: {detail}")
        return 1

    s = make_sample_voice(48000)
    cpu, _, _ = render_offline(s.data, 48000, DEFAULTS, 1024, CpuBackend().create_pipeline)
    gpu, _, _ = render_offline(s.data, 48000, DEFAULTS, 1024, RocmBackend().create_pipeline)
    diff = float(np.max(np.abs(cpu - gpu)))
    print(f"\nCPU vs ROCm max sample difference: {diff:.2e}")
    parity = diff < 1e-4

    rows = benchmark(CpuBackend(), DEFAULTS) + benchmark(RocmBackend(), DEFAULTS)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['backend']:5} chunk {r['chunk']:5}  p50 {r['p50_ms']:.3f} ms  p95 {r['p95_ms']:.3f} ms  RTF {r['rtf_p95']:.3f}  overruns {r['overruns']}")
    print(f"\nwrote {args.csv} and {args.json}")
    print("PASS" if parity else "FAIL: ROCm output differs from CPU")
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
