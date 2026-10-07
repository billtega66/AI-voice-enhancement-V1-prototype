"""Synthetic sample voice with labelled noise events (speech, keyboard, paper, fan).

Used by tests, the CLI benchmark and the self-test so that results never depend on a
microphone. Mirrors the event layout of makeSampleVoice() in web/src/dsp.js; the
random noise differs, which is fine because it is only test material.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .engine import biquad_coefs

DEFAULT_EVENTS = [
    {"type": "speech", "s": 0.6, "e": 2.8, "db": -24}, {"type": "keys", "s": 3.1, "e": 3.9, "db": -22},
    {"type": "speech", "s": 4.3, "e": 6.3, "db": -27}, {"type": "paper", "s": 6.7, "e": 7.25, "db": -36},
    {"type": "speech", "s": 7.7, "e": 9.6, "db": -38}, {"type": "keys", "s": 10.0, "e": 10.8, "db": -22},
    {"type": "speech", "s": 11.2, "e": 13.4, "db": -26},
]
VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240), (660, 1720, 2410)]


@dataclass
class Sample:
    data: np.ndarray
    fs: int
    events: list

    def label_at(self, t: float) -> str:
        for e in self.events:
            if e["s"] <= t < e["e"]:
                return e["type"]
        return "none"


def _bq(x, kind, fs, f0, q, g=0.0):
    c = biquad_coefs(kind, fs, f0, q, g)
    return lfilter([c[0], c[1], c[2]], [1.0, c[3], c[4]], x)


def make_sample_voice(fs: int = 48000, seconds: float = 14.0, events=None, noise_db: float = -52.0, f0_base: float = 135.0, seed: int = 1234) -> Sample:
    events = events or DEFAULT_EVENTS
    rng = np.random.default_rng(seed)
    n = int(round(fs * seconds))
    x = np.zeros(n)
    for ei, ev in enumerate(events):
        i0, i1 = int(round(ev["s"] * fs)), min(n, int(round(ev["e"] * fs)))
        if i1 - i0 < fs * 0.05:
            continue
        m = i1 - i0
        t = np.arange(m) / fs
        if ev["type"] == "speech":
            T = ev["s"] + t
            f0 = f0_base + 20 * np.sin(2 * np.pi * 0.7 * T) - 10 * (t / (ev["e"] - ev["s"])) + 4 * ei
            sp = t * 4.2
            si = np.floor(sp).astype(int)
            fr = sp - si
            env = np.maximum(0, np.sin(np.pi * fr)) ** 0.6
            v = np.zeros(m)
            phase0 = np.cumsum(2 * np.pi * f0 / fs)
            for k in range(1, 61):
                fk = k * f0
                alive = fk < 5000
                if not alive.any():
                    break
                a = np.zeros(m)
                for j in range(3):
                    F = np.array([VOWELS[(s + ei) % 5][j] for s in range(si.max() + 1)])[si]
                    a += np.exp(-(((fk - F) / (80 + 40 * j)) ** 2)) * (1 - 0.3 * j)
                v += np.where(alive, (a * 0.9 + 0.12 / k) * np.sin(k * phase0), 0)
            sib = _bq(rng.uniform(-1, 1, m), "bandpass", fs, 6500, 1.2)
            s_env = np.where((si % 3 == 1) & (fr < 0.3), np.sin(np.pi * fr / 0.3), 0)
            seg = v * env + sib * s_env * 9
        elif ev["type"] == "keys":
            seg = np.zeros(m)
            pos = 0
            while pos < m:
                L = min(m - pos, int(0.03 * fs))
                seg[pos:pos + L] = rng.uniform(-1, 1, L) * np.exp(-np.arange(L) / fs / 0.0025)
                pos += int(round((0.11 + 0.08 * rng.random()) * fs))
        else:  # paper
            seg = _bq(rng.uniform(-1, 1, m), "bandpass", fs, 3000, 0.6)
            am = np.repeat(np.abs(rng.uniform(-1, 1, m // 200 + 1)), 200)[:m]
            seg = seg * am * np.sin(np.pi * np.arange(m) / m)
        r = np.sqrt(np.mean(seg ** 2)) or 1.0
        x[i0:i1] += seg * (10 ** (ev["db"] / 20) / r)
    fan = _bq(rng.uniform(-1, 1, n), "lowpass", fs, 900, 0.7) + 0.15 * np.sin(2 * np.pi * 120 * np.arange(n) / fs)
    x += fan * (10 ** (noise_db / 20) / np.sqrt(np.mean(fan ** 2)))
    return Sample(x.astype(np.float32), fs, [dict(e) for e in events])
