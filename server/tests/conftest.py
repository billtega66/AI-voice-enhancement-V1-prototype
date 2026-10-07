import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from voice_engine.dsp import make_sample_voice

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js not installed (browser-engine parity tests)")


@pytest.fixture(scope="session")
def sample():
    return make_sample_voice(48000)


def run_node(script: str) -> str:
    """Run a snippet with the browser engine loaded (web/src/*.js) and return stdout."""
    pre = "".join(f"require({json.dumps(str(ROOT / 'web/src' / (f + '.js')))});" for f in ("profile", "dsp", "analysis", "ai"))
    return subprocess.run([NODE, "-e", pre + script], check=True, capture_output=True, text=True).stdout


def tone(f, sec, db=-12, fs=48000):
    t = np.arange(int(round(sec * fs))) / fs
    return (10 ** (db / 20) * np.sin(2 * np.pi * f * t)).astype(np.float32)


def level(x, s, e, fs=48000):
    seg = x[int(s * fs):int(e * fs)].astype(np.float64)
    return 20 * np.log10(max(np.sqrt(np.mean(seg ** 2)), 1e-10))
