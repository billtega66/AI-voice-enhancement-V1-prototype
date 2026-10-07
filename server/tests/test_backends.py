import sys
import types

import numpy as np
import pytest

from voice_engine.backends import CpuBackend, RocmBackend, available_backends, get_backend
from voice_engine.backends.rocm import TorchOps
from voice_engine.dsp import render_offline
from voice_engine.dsp.engine import NoiseSuppressor
from voice_engine.profile import DEFAULTS


def test_auto_picks_cpu_without_rocm(monkeypatch):
    monkeypatch.delenv("VOICE_BACKEND", raising=False)
    assert get_backend().name == ("rocm" if RocmBackend.available()[0] else "cpu")
    assert get_backend("cpu").name == "cpu"


def test_unknown_and_unavailable_backends():
    with pytest.raises(ValueError):
        get_backend("tpu")
    if not RocmBackend.available()[0]:
        with pytest.raises(RuntimeError, match="unavailable"):
            get_backend("rocm")
        with pytest.raises(RuntimeError):
            RocmBackend().create_pipeline(48000, DEFAULTS)


def test_backend_listing():
    rows = {r["name"]: r for r in available_backends()}
    assert rows["cpu"]["available"] is True and "detail" in rows["rocm"]


def _fake_torch():
    """Minimal numpy-backed stand-in for the torch API surface TorchOps uses.
    It lets CI check the GPU code path's call contract without an AMD GPU; it says
    nothing about ROCm itself (scripts/verify_rocm.py does that on real hardware)."""
    class T(np.ndarray):
        def detach(self): return self
        def to(self, *_): return self
        def numpy(self): return np.asarray(self)
    t = types.SimpleNamespace()
    t.device = lambda d: d
    t.float64 = np.float64
    wrap = lambda a: np.asarray(a).view(T)
    t.as_tensor = lambda a, device=None: wrap(np.array(a, dtype=np.float64))
    t.zeros = lambda n, dtype=None, device=None: wrap(np.zeros(n))
    t.ones = lambda n, dtype=None, device=None: wrap(np.ones(n))
    t.fft = types.SimpleNamespace(fft=lambda a: wrap(np.fft.fft(np.asarray(a))), ifft=lambda a: wrap(np.fft.ifft(np.asarray(a))))
    t.minimum = lambda a, b: wrap(np.minimum(a, b))
    t.where = lambda c, a, b: wrap(np.where(c, a, b))
    t.clamp_min = lambda a, v: wrap(np.maximum(a, v))
    t.flip = lambda a, dims: wrap(np.asarray(a)[::-1])
    t.cat = lambda xs: wrap(np.concatenate([np.asarray(x) for x in xs]))
    return t


def test_torch_ops_contract_matches_numpy(monkeypatch, sample):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    ops = TorchOps("cuda:0")
    a, b = NoiseSuppressor(48000), NoiseSuppressor(48000, ops=ops)
    x = sample.data[:48000]
    ya, yb = x.copy(), x.copy()
    for i in range(0, len(x), 1024):
        a.process(ya[i:i + 1024]); b.process(yb[i:i + 1024])
    assert np.max(np.abs(ya - yb)) < 1e-6
