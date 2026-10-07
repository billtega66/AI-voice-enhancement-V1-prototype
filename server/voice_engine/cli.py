"""voice-engine command line.

  voice-engine serve [--host 0.0.0.0 --port 8000 --backend auto]
  voice-engine enhance in.wav out.wav [--profile profile.json] [--prompt "warmer, podcast"]
  voice-engine analyze in.wav
  voice-engine interpret "make it warmer" [--profile profile.json]
  voice-engine benchmark [--backend cpu] [--chunks 256,512,1024] [--csv out.csv]
  voice-engine backends
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time

from .ai import get_interpreter
from .analysis import analyze_voice
from .audio_io import read_audio, wav_bytes
from .backends import available_backends, get_backend
from .dsp.engine import render_offline
from .metrics import benchmark
from .profile import DEFAULT_REFERENCE, DEFAULTS, full_profile


def load_env_file(path) -> int:
    """Minimal .env reader (KEY=VALUE, # comments). Existing environment variables win."""
    import os
    n = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v; n += 1
    return n


def _load_profile(path):
    if not path:
        return dict(DEFAULTS)
    obj = json.load(open(path))
    return full_profile(obj.get("params", obj))


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, float) and math.isfinite(v) else str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voice-engine", description="AI voice enhancement engine (CPU / AMD ROCm)")
    ap.add_argument("--env-file", default=None, help="read LLM_* and other settings from this file (default: .env or ../.env if present)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.add_argument("--backend", default=None)
    e = sub.add_parser("enhance"); e.add_argument("input"); e.add_argument("output"); e.add_argument("--profile"); e.add_argument("--prompt"); e.add_argument("--backend", default=None); e.add_argument("--chunk", type=int, default=1024)
    a = sub.add_parser("analyze"); a.add_argument("input")
    i = sub.add_parser("interpret"); i.add_argument("text"); i.add_argument("--profile")
    b = sub.add_parser("benchmark"); b.add_argument("--backend", default=None); b.add_argument("--chunks", default="256,512,1024,2048"); b.add_argument("--seconds", type=float, default=14.0); b.add_argument("--csv"); b.add_argument("--profile")
    sub.add_parser("backends")
    args = ap.parse_args(argv)
    import os
    env_file = args.env_file or next((p for p in (".env", "../.env") if os.path.exists(p)), None)
    if env_file:
        load_env_file(env_file)

    if args.cmd == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.backend), host=args.host, port=args.port)
        return 0

    if args.cmd == "backends":
        for row in available_backends():
            print(f"{row['name']:5}  {'yes' if row['available'] else 'no ':3}  {row['detail']}")
        return 0

    if args.cmd == "analyze":
        data, fs = read_audio(args.input)
        print(json.dumps(analyze_voice(data, fs), indent=2, default=lambda o: None))
        return 0

    if args.cmd == "interpret":
        from .ai import ModelUnavailable, OfflineInterpreter
        prof = _load_profile(args.profile)
        interp = get_interpreter()
        ctx = {"profile": prof, "analysis": None, "reference": DEFAULT_REFERENCE, "history": [], "owner": "cli"}
        try:
            res = interp.interpret(args.text, ctx)
        except ModelUnavailable as e:
            print(f"[{interp.name}] {e} Falling back to offline rules.", file=sys.stderr)
            interp = OfflineInterpreter(); res = interp.interpret(args.text, ctx)
        print(json.dumps({"interpreter": interp.name, "model": getattr(interp, "model", None), **res}, indent=2))
        return 0

    if args.cmd == "enhance":
        data, fs = read_audio(args.input)
        params = _load_profile(args.profile)
        if args.prompt:
            from .ai import ModelUnavailable, OfflineInterpreter
            interp = get_interpreter()
            ctx = {"profile": params, "analysis": analyze_voice(data, fs), "reference": DEFAULT_REFERENCE, "history": [], "owner": "cli"}
            try:
                res = interp.interpret(args.prompt, ctx)
            except ModelUnavailable as e:
                print(f"[{interp.name}] {e} Falling back to offline rules.", file=sys.stderr)
                interp = OfflineInterpreter(); res = interp.interpret(args.prompt, ctx)
            params = full_profile({**params, **res["changes"]})
            print(f"[{interp.name}] {res.get('reply')}", file=sys.stderr)
        backend = get_backend(args.backend)
        t0 = time.perf_counter()
        out, infos, lat = render_offline(data, fs, params, args.chunk, backend.create_pipeline)
        ms = (time.perf_counter() - t0) * 1000
        open(args.output, "wb").write(wav_bytes(out, fs))
        print(f"wrote {args.output}: {len(data) / fs:.1f} s in {ms:.0f} ms on {backend.name} (real-time factor {ms / (len(data) / fs * 1000):.3f})", file=sys.stderr)
        return 0

    if args.cmd == "benchmark":
        backend = get_backend(args.backend)
        rows = benchmark(backend, _load_profile(args.profile), 48000, tuple(int(c) for c in args.chunks.split(",")), args.seconds)
        cols = ["backend", "chunk", "budget_ms", "mean_ms", "p50_ms", "p95_ms", "rtf_p95", "overruns", "chunks", "pipeline_latency_ms", "est_end_to_end_ms"]
        print("  ".join(f"{c:>10}" for c in cols))
        for r in rows:
            print("  ".join(f"{_fmt(r[c]):>10}" for c in cols))
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
