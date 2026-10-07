"""Real-time voice enhancement DSP engine (server CPU backend).

This is a sample-accurate port of ``web/src/dsp.js`` so that the browser and the
server produce the same audio for the same Voice Profile. ``tests/test_parity.py``
checks that. Every stage keeps its own state, processes float32 blocks in place,
and gives the same result for any block size.

Per-sample recursions (filters, envelopes, gate, limiter) are compiled with
numba. The STFT noise suppressor is vectorised with numpy per frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from ..profile import DEFAULTS

# --------------------------------------------------------------------------- helpers


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float) -> float:
    return 20.0 * math.log10(max(abs(x), 1e-10))


def pow_to_db(p: float) -> float:
    return 10.0 * math.log10(max(p, 1e-20))


def clamp(v, lo, hi):
    return min(hi, max(lo, v))


def peak(b: np.ndarray) -> float:
    return float(np.max(np.abs(b))) if b.size else 0.0


# --------------------------------------------------------------------------- biquad


def biquad_coefs(kind: str, fs: float, f0: float, q: float, gain_db: float = 0.0) -> np.ndarray:
    """RBJ cookbook coefficients [b0, b1, b2, a1, a2] normalised by a0 (same maths as dsp.js)."""
    f0 = min(f0, fs * 0.45)
    A = 10.0 ** (gain_db / 40.0)
    w = 2.0 * math.pi * f0 / fs
    c, s = math.cos(w), math.sin(w)
    al = s / (2.0 * q)
    sq = 2.0 * math.sqrt(A) * al
    if kind == "highpass":
        b0 = (1 + c) / 2; b1 = -(1 + c); b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al
    elif kind == "lowpass":
        b0 = (1 - c) / 2; b1 = 1 - c; b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al
    elif kind == "bandpass":
        b0 = al; b1 = 0.0; b2 = -al; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al
    elif kind == "peaking":
        b0 = 1 + al * A; b1 = -2 * c; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * c; a2 = 1 - al / A
    elif kind == "lowshelf":
        b0 = A * ((A + 1) - (A - 1) * c + sq); b1 = 2 * A * ((A - 1) - (A + 1) * c); b2 = A * ((A + 1) - (A - 1) * c - sq)
        a0 = (A + 1) + (A - 1) * c + sq; a1 = -2 * ((A - 1) + (A + 1) * c); a2 = (A + 1) + (A - 1) * c - sq
    elif kind == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * c + sq); b1 = -2 * A * ((A - 1) + (A + 1) * c); b2 = A * ((A + 1) + (A - 1) * c - sq)
        a0 = (A + 1) - (A - 1) * c + sq; a1 = 2 * ((A - 1) - (A + 1) * c); a2 = (A + 1) - (A - 1) * c - sq
    else:
        raise ValueError(f"unknown biquad type {kind}")
    return np.array([b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0], dtype=np.float64)


@njit(cache=True, fastmath=False)
def _bq_tick(x, c, s):
    y = c[0] * x + c[1] * s[0] + c[2] * s[1] - c[3] * s[2] - c[4] * s[3]
    if -1e-25 < y < 1e-25:
        y = 0.0
    s[1] = s[0]; s[0] = x; s[3] = s[2]; s[2] = y
    return y


@njit(cache=True)
def _bq_process(buf, c, s):
    for i in range(buf.shape[0]):
        buf[i] = _bq_tick(np.float64(buf[i]), c, s)


class Biquad:
    """State layout: [x1, x2, y1, y2] (float64, like the JS object fields)."""

    def __init__(self):
        self.c = np.array([1.0, 0, 0, 0, 0])
        self.s = np.zeros(4)

    def set(self, kind, fs, f0, q, gain_db=0.0):
        self.c = biquad_coefs(kind, fs, f0, q, gain_db)
        return self

    def reset(self):
        self.s[:] = 0

    def process(self, b):
        _bq_process(b, self.c, self.s)


# --------------------------------------------------------------------------- noise suppression


class NumpyOps:
    """Array backend for the STFT stage. RocmOps (backends/rocm.py) implements the same
    interface with torch on an AMD GPU, so the identical algorithm runs on either device."""
    name = "numpy"

    def from_host(self, a): return np.asarray(a, dtype=np.float64)
    def to_host(self, a): return np.asarray(a, dtype=np.float64)
    def zeros(self, n): return np.zeros(n)
    def ones(self, n): return np.ones(n)
    def fft(self, a): return np.fft.fft(a)
    def ifft(self, a): return np.fft.ifft(a)
    def real(self, a): return a.real
    def imag(self, a): return a.imag
    def minimum(self, a, b): return np.minimum(a, b)
    def where(self, c, a, b): return np.where(c, a, b)
    def clamp_min(self, a, v): return np.maximum(a, v)
    def flip(self, a): return a[::-1]
    def concat(self, a, b): return np.concatenate([a, b])
    def dot(self, a): return float(np.dot(a, a))


NUMPY_OPS = NumpyOps()


class NoiseSuppressor:
    """STFT spectral suppression with a minimum-tracking noise estimate. Latency = N samples."""

    def __init__(self, fs: float, N: int = 512, ops=None):
        self.fs, self.N, self.hop = fs, N, N // 2
        self.ops = ops or NUMPY_OPS
        i = np.arange(N)
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * i / N)).astype(np.float32)
        self.win64 = self.win.astype(np.float64)
        self.rise = 10 ** (3 * self.hop / fs / 10)
        self.amount = 0.6
        self.reset()

    @property
    def latency(self) -> int:
        return self.N

    def reset(self):
        B, o = self.N // 2 + 1, self.ops
        self.in_buf = np.zeros(self.N, np.float32)
        self.acc = np.zeros(self.N, np.float32)
        self.ready = np.zeros(self.hop, np.float32)
        self.k = 0
        self.frames = 0
        self.noise, self.ps, self.g_prev = o.zeros(B), o.zeros(B), o.ones(B)
        self.in_e = 0.0
        self.out_e = 0.0
        self.reduction_db = 0.0

    def process(self, b: np.ndarray):
        hop, i, n = self.hop, 0, b.shape[0]
        while i < n:
            m = min(hop - self.k, n - i)
            x = b[i:i + m].copy()
            self.in_buf[hop + self.k: hop + self.k + m] = x
            b[i:i + m] = self.ready[self.k:self.k + m]
            self.k += m
            i += m
            if self.k == hop:
                self.k = 0
                self._frame()

    def _frame(self):
        N, hop, B, o = self.N, self.hop, self.N // 2 + 1, self.ops
        frame = self.in_buf.astype(np.float64) * self.win64
        ein = float(np.dot(frame, frame))
        X = o.fft(o.from_host(frame))
        a = clamp(self.amount, 0.0, 1.0)
        alpha, gmin = 1 + 2 * a, db_to_lin(-30 * a)
        P = o.real(X)[:B] ** 2 + o.imag(X)[:B] ** 2
        if self.frames < 4:
            self.ps = P
            if self.frames == 0:
                self.noise = self.ps
            else:
                mn = o.minimum(self.noise, self.ps)
                self.noise = o.where(mn == 0, self.ps, mn)
        else:
            self.ps = 0.7 * self.ps + 0.3 * P
            self.noise = o.where(self.ps < self.noise, self.noise + 0.25 * (self.ps - self.noise), self.noise * self.rise)
        g = 1 - alpha * (self.noise * 1.4) / (P + 1e-20)
        g = o.clamp_min(g, gmin)
        g = 0.35 * self.g_prev + 0.65 * g
        if a == 0:
            g = o.ones(B)
        self.g_prev = g
        full = o.concat(g, o.flip(g[1:N // 2]))
        y = o.to_host(o.real(o.ifft(X * full))) * self.win64
        self.acc += y.astype(np.float32)
        eout = float(np.dot(y, y))
        self.ready[:] = self.acc[:hop]
        self.acc[:N - hop] = self.acc[hop:]
        self.acc[N - hop:] = 0
        self.in_buf[:N - hop] = self.in_buf[hop:]
        self.frames += 1
        self.in_e = 0.8 * self.in_e + 0.2 * ein
        self.out_e = 0.8 * self.out_e + 0.2 * eout
        self.reduction_db = min(0.0, pow_to_db(self.out_e + 1e-20) - pow_to_db(self.in_e + 1e-20))


# --------------------------------------------------------------------------- speech gate (VAD)

# scalar state indices for the gate kernel
_W, _DEC, _HOPC, _T, _FLOOR, _HANG, _SPEECH, _ACTIVE, _VOICED, _VRUN, _LASTV, _PER, _F0, _LVL, _G, _DW, _FRAMES, _HASFLOOR = range(18)


@njit(cache=True)
def _gate_analyze(st, ring, f, frame_len, hop, fs, dfs, min_lag, max_lag, sens, hold_ms):
    L = frame_len
    start = (int(st[_W]) - L) & 2047
    e = 0.0
    for i in range(L):
        v = ring[(start + i) & 2047]
        f[i] = v
        e += np.float64(v) * np.float64(v)
    p = e / L
    lvl = 10.0 * math.log10(max(p, 1e-20))
    if not np.isfinite(lvl):
        lvl = -120.0
    st[_LVL] = lvl
    st[_T] += hop / fs
    st[_FRAMES] += 1
    if st[_HASFLOOR] == 0:
        st[_FLOOR] = lvl
        st[_HASFLOOR] = 1
    if lvl < st[_FLOOR]:
        st[_FLOOR] += (lvl - st[_FLOOR]) * 0.3
    else:
        st[_FLOOR] += min(lvl - st[_FLOOR], 2 * hop / fs)
    best = 0.0
    best_lag = 0
    if lvl > -75:
        for lag in range(min_lag, max_lag + 1):
            s = 0.0; e1 = 0.0; e2 = 0.0
            for i in range(L - lag):
                a = np.float64(f[i]); c = np.float64(f[i + lag])
                s += a * c; e1 += a * a; e2 += c * c
            rr = s / math.sqrt(e1 * e2 + 1e-20)
            if rr > best:
                best = rr; best_lag = lag
    sens = min(1.0, max(0.0, sens))
    thr_e = 18 - 14 * sens
    thr_p = 0.75 - 0.4 * sens
    abs_min = -50 - 15 * sens
    energetic = lvl > st[_FLOOR] + thr_e and lvl > abs_min
    periodic = best > thr_p and energetic
    f0 = dfs / best_lag if best_lag else 0.0
    if periodic and st[_VRUN] > 0 and st[_F0] > 0 and abs(math.log2(f0 / st[_F0])) > 0.25:
        st[_VRUN] = 1
    else:
        st[_VRUN] = st[_VRUN] + 1 if periodic else 0
    st[_PER] = best
    if periodic:
        st[_F0] = f0
    voiced = st[_VRUN] >= 3
    st[_VOICED] = 1.0 if voiced else 0.0
    if voiced:
        st[_LASTV] = st[_T]
    active = energetic and (voiced or st[_T] - st[_LASTV] < 0.15)
    st[_ACTIVE] = 1.0 if active else 0.0
    if active:
        st[_HANG] = hold_ms / 1000.0
    else:
        st[_HANG] -= hop / fs
    st[_SPEECH] = 1.0 if (active or st[_HANG] > 0) else 0.0


@njit(cache=True)
def _gate_process(b, apply_gate, st, ring, f, delay, mask, c1, s1, c2, s2, dec, frame_len, hop, fs, dfs,
                  min_lag, max_lag, sens, hold_ms, target0, att, rel, L):
    for i in range(b.shape[0]):
        x = np.float64(b[i])
        y = _bq_tick(_bq_tick(x, c1, s1), c2, s2)
        st[_DEC] += 1
        if st[_DEC] >= dec:
            st[_DEC] = 0
            ring[int(st[_W])] = y
            st[_W] = (int(st[_W]) + 1) & 2047
        st[_HOPC] += 1
        if st[_HOPC] >= hop:
            st[_HOPC] = 0
            _gate_analyze(st, ring, f, frame_len, hop, fs, dfs, min_lag, max_lag, sens, hold_ms)
        mask[i] = 1 if st[_SPEECH] > 0 else 0
        if apply_gate:
            target = 1.0 if st[_SPEECH] > 0 else target0
            g = st[_G]
            g = target + (g - target) * (att if target > g else rel)
            st[_G] = g
            dw = int(st[_DW])
            delay[dw] = x
            dw = 0 if dw == L else dw + 1
            st[_DW] = dw
            b[i] = np.float64(delay[dw]) * g


class SpeechGate:
    """Speech detector (pitch periodicity + energy over an adaptive floor) with a non-speech mute."""

    def __init__(self, fs: float):
        self.fs = fs
        self.dec = max(1, round(fs / 12000))
        self.dfs = fs / self.dec
        self.lp1 = Biquad().set("lowpass", fs, 0.42 * self.dfs, 0.707)
        self.lp2 = Biquad().set("lowpass", fs, 0.42 * self.dfs, 0.707)
        self.frame_len = _js_round(0.03 * self.dfs)
        self.hop = _js_round(0.01 * fs)
        self.min_lag = math.floor(self.dfs / 400)
        self.max_lag = math.ceil(self.dfs / 70)
        self.lookahead = _js_round(0.03 * fs)
        self.att = math.exp(-1 / (0.004 * fs))
        self.rel = math.exp(-1 / (0.08 * fs))
        self.sensitivity, self.hold_ms, self.attenuation_db = 0.5, 300.0, -30.0
        self.mask = np.zeros(0, np.uint8)
        self.reset()

    def reset(self):
        self.ring = np.zeros(2048, np.float32)
        self.f = np.zeros(self.frame_len, np.float32)
        self.delay = np.zeros(self.lookahead + 1, np.float32)
        self.st = np.zeros(18)
        self.st[_LASTV] = -10.0
        self.st[_LVL] = -120.0
        self.st[_G] = 1.0
        self.lp1.reset(); self.lp2.reset()

    speech = property(lambda s: bool(s.st[_SPEECH]))
    active = property(lambda s: bool(s.st[_ACTIVE]))
    voiced = property(lambda s: bool(s.st[_VOICED]))
    f0 = property(lambda s: float(s.st[_F0]))
    periodicity = property(lambda s: float(s.st[_PER]))
    level_db = property(lambda s: float(s.st[_LVL]))
    g = property(lambda s: float(s.st[_G]))

    def process(self, b: np.ndarray, apply_gate: bool):
        if self.mask.shape[0] != b.shape[0]:
            self.mask = np.zeros(b.shape[0], np.uint8)
        _gate_process(b, apply_gate, self.st, self.ring, self.f, self.delay, self.mask,
                      self.lp1.c, self.lp1.s, self.lp2.c, self.lp2.s, self.dec, self.frame_len, self.hop,
                      float(self.fs), float(self.dfs), self.min_lag, self.max_lag, float(self.sensitivity),
                      float(self.hold_ms), db_to_lin(self.attenuation_db), self.att, self.rel, self.lookahead)


def _js_round(x: float) -> int:
    """JavaScript Math.round (half up), not Python's banker's rounding."""
    return int(math.floor(x + 0.5))


# --------------------------------------------------------------------------- de-esser, compressor, limiter


@njit(cache=True)
def _deess(b, cb, sb, cd, sd, st, att, rel, thr, max_db):
    min_g = 0.0
    for i in range(b.shape[0]):
        x = np.float64(b[i])
        band = _bq_tick(x, cb, sb)
        d = _bq_tick(x, cd, sd)
        a = -d if d < 0 else d
        env = st[0]
        env = a + (env - a) * att if a > env else a + (env - a) * rel
        st[0] = env
        over = 20.0 * math.log10(max(abs(env * 1.414), 1e-10)) - thr
        red = min(max_db, over * 0.75) if over > 0 else 0.0
        g = 10.0 ** (-red / 20.0)
        b[i] = x - (1 - g) * band
        if -red < min_g:
            min_g = -red
    return min_g


class DeEsser:
    def __init__(self, fs):
        self.fs = fs
        self.bp, self.det = Biquad(), Biquad()
        self.att = math.exp(-1 / (0.001 * fs)); self.rel = math.exp(-1 / (0.06 * fs))
        self.set(6500, -32, 6)
        self.reset()

    def set(self, freq, thr, max_db):
        self.freq, self.thr, self.max_db = freq, thr, max_db
        self.bp.set("bandpass", self.fs, freq, 1.4); self.det.set("bandpass", self.fs, freq, 1.4)

    def reset(self):
        self.bp.reset(); self.det.reset(); self.st = np.zeros(1); self.gr_db = 0.0

    def process(self, b):
        self.gr_db = _deess(b, self.bp.c, self.bp.s, self.det.c, self.det.s, self.st, self.att, self.rel, float(self.thr), float(self.max_db))


@njit(cache=True)
def _comp(b, st, thr, ratio, knee, a_att, a_rel):
    min_g = 0.0
    s = 1.0 / ratio - 1.0
    for i in range(b.shape[0]):
        lvl = 20.0 * math.log10(max(abs(np.float64(b[i])), 1e-10))
        env = st[0]
        env = lvl + (env - lvl) * a_att if lvl > env else lvl + (env - lvl) * a_rel
        st[0] = env
        over = env - thr
        if over <= -knee / 2:
            g = 0.0
        elif over < knee / 2:
            g = s * (over + knee / 2) ** 2 / (2 * knee)
        else:
            g = s * over
        b[i] = np.float64(b[i]) * 10.0 ** (g / 20.0)
        if g < min_g:
            min_g = g
    return min_g


class Compressor:
    def __init__(self, fs):
        self.fs, self.knee = fs, 6.0
        self.set(-26, 2.5, 8, 160)
        self.reset()

    def set(self, thr, ratio, att_ms, rel_ms):
        self.thr, self.ratio = float(thr), max(1.0, float(ratio))
        self.a_att = math.exp(-1 / (max(0.1, att_ms) / 1000 * self.fs))
        self.a_rel = math.exp(-1 / (max(1.0, rel_ms) / 1000 * self.fs))

    def reset(self):
        self.st = np.array([-120.0]); self.gr_db = 0.0

    def process(self, b):
        self.gr_db = _comp(b, self.st, self.thr, self.ratio, self.knee, self.a_att, self.a_rel)


@njit(cache=True)
def _limit(b, st, ceil, rel):
    g = st[0]
    min_g = 1.0
    for i in range(b.shape[0]):
        x = np.float64(b[i])
        a = abs(x)
        need = ceil / a if a > ceil else 1.0
        g = min(need, 1 - (1 - g) * rel)
        y = x * g
        if y > ceil:
            y = ceil
        elif y < -ceil:
            y = -ceil
        b[i] = y
        if g < min_g:
            min_g = g
    st[0] = g
    return min_g


class Limiter:
    def __init__(self, fs):
        self.rel = math.exp(-1 / (0.06 * fs)); self.ceiling_db = -1.0; self.reset()

    def reset(self):
        self.st = np.array([1.0]); self.gr_db = 0.0

    def process(self, b):
        self.gr_db = lin_to_db(_limit(b, self.st, db_to_lin(self.ceiling_db), self.rel))


# --------------------------------------------------------------------------- loudness


def k_filters(fs):
    return [Biquad().set("highshelf", fs, 1681.97, 0.7071, 4.0), Biquad().set("highpass", fs, 38.13, 0.5)]


@njit(cache=True)
def _loud(b, mask, use_mask, c0, s0, c1, s1, st, F, dt, a, target, smooth):
    # st: [ms, started, acc, n, sp, est, has_est, gain_db, cur, speech_t]
    for i in range(b.shape[0]):
        y = _bq_tick(_bq_tick(np.float64(b[i]), c0, s0), c1, s1)
        st[2] += y * y
        st[4] += mask[i] if use_mask else 1
        st[3] += 1
        if st[3] == F:
            p = st[2] / F
            if st[1] == 0:
                st[1] = 1.0 if p > 1e-12 else 0.0
                st[0] = p
            else:
                st[0] += (p - st[0]) * a
            m = -0.691 + 10.0 * math.log10(max(st[0], 1e-20))
            if st[4] > F / 2 and m > -70:
                st[9] += dt
                tau = 0.5 if st[9] < 1.5 else 2.5
                if st[6] == 0:
                    st[5] = m; st[6] = 1
                else:
                    st[5] += (m - st[5]) * min(1.0, dt / tau)
            if st[6] == 1:
                want = min(24.0, max(-12.0, target - st[5]))
                st[7] += (want - st[7]) * min(1.0, dt / 0.8)
            st[2] = 0; st[3] = 0; st[4] = 0
        tgt = 10.0 ** (st[7] / 20.0)
        st[8] = tgt + (st[8] - tgt) * smooth
        b[i] = np.float64(b[i]) * st[8]


class LoudnessNormalizer:
    """Speech-gated normaliser on fixed 256-sample frames (block-size independent)."""

    def __init__(self, fs):
        self.fs, self.k, self.F, self.target = fs, k_filters(fs), 256, -16.0
        self.smooth = math.exp(-1 / (0.02 * fs))
        self.reset()

    def reset(self):
        for f in self.k:
            f.reset()
        self.st = np.zeros(10); self.st[8] = 1.0

    @property
    def gain_db(self):
        return float(self.st[7])

    def process(self, b, mask=None):
        F, fs = self.F, self.fs
        dt = F / fs
        a = 1 - math.exp(-dt / 0.4)
        m = mask if mask is not None else np.ones(b.shape[0], np.uint8)
        _loud(b, m, mask is not None, self.k[0].c, self.k[0].s, self.k[1].c, self.k[1].s, self.st, F, dt, a, float(self.target), self.smooth)


class LoudnessMeter:
    """Momentary-style meter (400 ms EMA of K-weighted power). For display only."""

    def __init__(self, fs):
        self.fs, self.k, self.ms, self.started = fs, k_filters(fs), 0.0, False

    def push(self, b) -> float:
        y = b.astype(np.float32).copy()
        self.k[0].process(y); self.k[1].process(y)
        cur = float(np.mean(y.astype(np.float64) ** 2)) if y.size else 0.0
        a = 1 - math.exp(-(b.shape[0] / self.fs) / 0.4)
        if not self.started:
            self.started, self.ms = cur > 1e-12, cur
        else:
            self.ms += (cur - self.ms) * a
        return -0.691 + pow_to_db(self.ms)


def integrated_lufs(data: np.ndarray, fs: float):
    """ITU-R BS.1770 style integrated loudness with absolute (-70) and relative (-10) gates."""
    y = data.astype(np.float32).copy()
    k = k_filters(fs)
    k[0].process(y); k[1].process(y)
    y = y.astype(np.float64)
    W, H = _js_round(0.4 * fs), _js_round(0.1 * fs)
    blocks = np.array([np.mean(y[i:i + W] ** 2) for i in range(0, len(y) - W + 1, H)]) if len(y) >= W else np.zeros(0)
    L = -0.691 + 10 * np.log10(np.maximum(blocks, 1e-20))
    keep = blocks[L > -70]
    if keep.size == 0:
        return -math.inf, L
    rel = keep[(-0.691 + 10 * np.log10(np.maximum(keep, 1e-20))) > (-0.691 + 10 * math.log10(keep.mean())) - 10]
    return -0.691 + 10 * math.log10(rel.mean()), L


# --------------------------------------------------------------------------- pitch


@njit(cache=True)
def _pitch(b, buf, st, W, ratio, mask):
    # st: [w, p]
    dp = (1 - ratio) / W
    p = st[1]
    w = int(st[0])
    for i in range(b.shape[0]):
        buf[w] = b[i]
        p2 = p + 0.5
        if p2 >= 1:
            p2 -= 1
        out = 0.0
        for pp, gg in ((p, 1 - abs(2 * p - 1)), (p2, 1 - abs(2 * p2 - 1))):
            pos = w - (1 + pp * W)
            ii = math.floor(pos)
            fr = pos - ii
            a = np.float64(buf[int(ii) & mask]); c = np.float64(buf[(int(ii) + 1) & mask])
            out += gg * (a + (c - a) * fr)
        b[i] = out
        p += dp
        p -= math.floor(p)
        w = (w + 1) & mask
    st[0] = w; st[1] = p


class PitchShifter:
    def __init__(self, fs, window_ms=40):
        self.W = _js_round(window_ms / 1000 * fs)
        size = 1
        while size < self.W * 2 + 8:
            size <<= 1
        self.mask = size - 1
        self.buf = np.zeros(size, np.float32)
        self.set_semitones(0)
        self.reset()

    def reset(self):
        self.buf[:] = 0; self.st = np.zeros(2)

    def set_semitones(self, st):
        self.semitones, self.ratio = st, 2 ** (st / 12)

    @property
    def latency(self):
        return _js_round(self.W / 2)

    def process(self, b):
        _pitch(b, self.buf, self.st, self.W, self.ratio, self.mask)


# --------------------------------------------------------------------------- pipeline

STAGES = ["input", "ns", "vad", "hpf", "eq", "pitch", "deess", "comp", "loud", "out"]


@dataclass
class BlockInfo:
    bypass: bool
    speech: bool
    voiced: bool
    f0: float
    periodicity: float
    in_db: float
    in_lufs: float
    out_db: float
    out_lufs: float
    ns_db: float = 0.0
    gate_db: float = 0.0
    deess_db: float = 0.0
    comp_db: float = 0.0
    lim_db: float = 0.0
    loud_db: float = 0.0

    def to_dict(self):
        """Same keys as the browser engine's info object."""
        return {"bypass": self.bypass, "speech": self.speech, "voiced": self.voiced, "f0": self.f0,
                "periodicity": self.periodicity, "inDb": self.in_db, "inLufs": self.in_lufs, "outDb": self.out_db,
                "outLufs": self.out_lufs, "nsDb": self.ns_db, "gateDb": self.gate_db, "deEssDb": self.deess_db,
                "compDb": self.comp_db, "limDb": self.lim_db, "loudDb": self.loud_db}


class Pipeline:
    """Input gain -> NS -> VAD/mute -> HPF -> EQ -> pitch -> de-esser -> compressor -> loudness -> output gain -> limiter."""

    backend_name = "cpu"

    def __init__(self, fs: float, params: dict | None = None, ops=None):
        self.fs = fs
        self.ns = NoiseSuppressor(fs, ops=ops)
        self.gate = SpeechGate(fs)
        self.hpf, self.mud, self.warm, self.pres, self.air = (Biquad() for _ in range(5))
        self.shifter = PitchShifter(fs)
        self.deess = DeEsser(fs)
        self.comp = Compressor(fs)
        self.loud = LoudnessNormalizer(fs)
        self.lim = Limiter(fs)
        self.in_meter, self.out_meter = LoudnessMeter(fs), LoudnessMeter(fs)
        self.p = dict(DEFAULTS)
        self.configure(params or {})

    def configure(self, params: dict):
        prev = self.p
        p = self.p = {**self.p, **params}
        fs = self.fs
        self.ns.amount = p["nsAmount"]
        self.gate.sensitivity, self.gate.hold_ms, self.gate.attenuation_db = p["vadSensitivity"], p["vadHoldMs"], p["vadAttenuationDb"]
        self.hpf.set("highpass", fs, p["hpfHz"], 0.707)
        self.mud.set("peaking", fs, 350, 1.0, p["mudDb"])
        self.warm.set("peaking", fs, 180, 0.8, p["warmthDb"])
        self.pres.set("peaking", fs, 4000, 0.9, p["presenceDb"])
        self.air.set("highshelf", fs, 10000, 0.707, p["airDb"])
        if p["pitchSemitones"] != prev["pitchSemitones"]:
            if abs(p["pitchSemitones"]) < 0.01:
                self.shifter.reset()
            self.shifter.set_semitones(p["pitchSemitones"])
        self.deess.set(p["deEssFreqHz"], p["deEssThresholdDb"], p["deEssMaxDb"])
        self.comp.set(p["compThresholdDb"], p["compRatio"], p["compAttackMs"], p["compReleaseMs"])
        self.loud.target = p["targetLufs"]
        self.lim.ceiling_db = p["limiterCeilingDb"]
        return self

    @property
    def pitch_active(self):
        return abs(self.p["pitchSemitones"]) >= 0.01

    @property
    def latency_samples(self) -> int:
        p = self.p
        return (self.ns.latency if p["nsEnabled"] else 0) + (self.gate.lookahead if p["vadEnabled"] else 0) + (self.shifter.latency if self.pitch_active else 0)

    @staticmethod
    def _gain(b, db):
        if db != 0:
            b[:] = (b.astype(np.float64) * db_to_lin(db)).astype(np.float32)  # float64 maths, float32 storage (as in JS)

    def process(self, b: np.ndarray, bypass: bool = False) -> BlockInfo:
        """Process one float32 mono block in place."""
        assert b.dtype == np.float32, "blocks must be float32"
        p = self.p
        in_pk = peak(b)
        in_m = self.in_meter.push(b)
        if bypass:
            self.gate.process(b.copy(), False)
            g = self.gate
            return BlockInfo(True, g.speech, g.voiced, g.f0, g.periodicity, lin_to_db(in_pk), in_m, lin_to_db(in_pk), self.out_meter.push(b))
        self._gain(b, p["inputGainDb"])
        if p["nsEnabled"]:
            self.ns.process(b)
        self.gate.process(b, bool(p["vadEnabled"]))
        speech = self.gate.speech
        if p["hpfEnabled"]:
            self.hpf.process(b)
        if p["mudDb"] != 0: self.mud.process(b)
        if p["warmthDb"] != 0: self.warm.process(b)
        if p["presenceDb"] != 0: self.pres.process(b)
        if p["airDb"] != 0: self.air.process(b)
        if self.pitch_active:
            self.shifter.process(b)
        if p["deEssEnabled"]:
            self.deess.process(b)
        else:
            self.deess.gr_db = 0.0
        if p["compEnabled"]:
            self.comp.process(b)
            self._gain(b, p["makeupDb"])
        else:
            self.comp.gr_db = 0.0
        if p["loudEnabled"]:
            self.loud.process(b, self.gate.mask)
        self._gain(b, p["outputGainDb"])
        if p["limiterEnabled"]:
            self.lim.process(b)
        else:
            self.lim.gr_db = 0.0
        return BlockInfo(False, speech, self.gate.voiced, self.gate.f0, self.gate.periodicity, lin_to_db(in_pk), in_m,
                         lin_to_db(peak(b)), self.out_meter.push(b),
                         ns_db=self.ns.reduction_db if p["nsEnabled"] else 0.0,
                         gate_db=lin_to_db(self.gate.g) if p["vadEnabled"] else 0.0,
                         deess_db=self.deess.gr_db, comp_db=self.comp.gr_db, lim_db=self.lim.gr_db,
                         loud_db=self.loud.gain_db if p["loudEnabled"] else 0.0)


def render_offline(data: np.ndarray, fs: float, params: dict, chunk: int = 1024, pipeline_factory=None):
    """Process a whole buffer and return (output aligned with input, list of BlockInfo, latency)."""
    data = np.asarray(data, dtype=np.float32)
    pl = pipeline_factory(fs, params) if pipeline_factory else Pipeline(fs, params)
    lat = pl.latency_samples
    total = data.shape[0] + lat
    out = np.zeros(total, np.float32)
    infos = []
    for i in range(0, total, chunk):
        n = min(chunk, total - i)
        b = np.zeros(n, np.float32)
        if i < data.shape[0]:
            seg = data[i:min(data.shape[0], i + n)]
            b[:seg.shape[0]] = seg
        info = pl.process(b)
        out[i:i + n] = b
        if i < data.shape[0]:
            infos.append(info)
    return out[lat:lat + data.shape[0]].copy(), infos, lat
