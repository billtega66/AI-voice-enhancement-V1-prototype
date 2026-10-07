"""Voice analysis (port of web/src/analysis.js): measures the user's own voice so the
profile can be personalised instead of forcing every speaker to the same settings."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .dsp.engine import SpeechGate, biquad_coefs, integrated_lufs, lin_to_db, pow_to_db


def _q(arr, p):
    return arr[min(len(arr) - 1, max(0, int(math.floor(p * (len(arr) - 1) + 0.5))))]


def _band(spec, fs, N, lo, hi):
    a, b = max(1, math.floor(lo * N / fs)), min(N // 2, math.ceil(hi * N / fs))
    return float(spec[a:b + 1].sum())


def analyze_voice(data: np.ndarray, fs: float) -> dict:
    data = np.asarray(data, dtype=np.float32)
    gate = SpeechGate(fs)
    hop = gate.hop
    frames = []
    for i in range(0, len(data) - hop + 1, hop):
        blk = data[i:i + hop].copy()
        gate.process(blk, False)
        seg = data[i:i + hop].astype(np.float64)
        frames.append({"t": i / fs, "speech": gate.speech, "voiced": gate.voiced, "f0": gate.f0 if gate.voiced else 0.0,
                       "r": gate.periodicity, "db": lin_to_db(math.sqrt(float(np.mean(seg ** 2))))})
    speech = [f for f in frames if f["speech"]]
    voiced = [f for f in frames if f["voiced"] and f["f0"] > 0]
    speech_s, dur = len(speech) * hop / fs, len(data) / fs
    if speech_s < 0.8:
        return {"ok": False, "reason": "Not enough speech was detected. Record at least a few seconds of talking.", "durationS": dur, "speechS": speech_s}

    f0s = sorted(f["f0"] for f in voiced)
    rs = [min(0.995, f["r"]) for f in voiced]
    r_mean = sum(rs) / len(rs) if rs else None

    N = 2048
    spec = np.zeros(N // 2 + 1)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
    for k in range(0, len(speech), 2):
        i0 = int(round(speech[k]["t"] * fs))
        if i0 + N > len(data):
            break
        spec += np.abs(np.fft.rfft(data[i0:i0 + N].astype(np.float64) * w)) ** 2
    total = _band(spec, fs, N, 100, 8000) or 1e-20
    rel = lambda lo, hi: pow_to_db(_band(spec, fs, N, lo, hi) / total)
    freqs = np.arange(N // 2 + 1) * fs / N
    m = (freqs >= 100) & (freqs <= 8000) & (np.arange(N // 2 + 1) >= 1)
    centroid = float((freqs[m] * spec[m]).sum() / spec[m].sum()) if spec[m].sum() > 0 else None
    pts, f = [], 125.0
    while f <= 8000:
        e = _band(spec, fs, N, f / 1.19, f * 1.19)
        if e > 0:
            pts.append((math.log2(f), pow_to_db(e)))
        f *= math.sqrt(2)
    slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) > 3 else None

    idx = np.concatenate([np.arange(int(round(f["t"] * fs)), min(len(data), int(round(f["t"] * fs)) + hop)) for f in speech])
    rms_db = pow_to_db(float(np.mean(data[idx].astype(np.float64) ** 2)))
    pk = float(np.max(np.abs(data)))
    integ, blocks = integrated_lufs(data, fs)
    sb = sorted(l for i, l in enumerate(blocks)
                if l > -70 and frames[min(len(frames) - 1, int(round((i * 0.1 + 0.2) * fs / hop)))]["speech"])
    lra = _q(sb, 0.95) - _q(sb, 0.10) if len(sb) > 4 else None
    quiet = sorted(f["db"] for f in frames if not f["speech"])
    noise_floor = _q(quiet, 0.5) if quiet else None

    c = biquad_coefs("bandpass", fs, 6500, 1.2)
    sib = lfilter([c[0], c[1], c[2]], [1, c[3], c[4]], data.astype(np.float64))
    sib_lv = sorted(pow_to_db(float(np.mean(sib[int(round(f["t"] * fs)):int(round(f["t"] * fs)) + hop] ** 2))) for f in speech)

    env = [f["db"] if f["speech"] else -120 for f in frames]
    syll, last = 0, -1
    for i in range(2, len(env) - 2):
        e = env[i]
        if e > -60 and e >= max(env[i - 2], env[i - 1], env[i + 1], env[i + 2]) and i - last >= 10:
            lo = min(env[max(0, i - 12):i]); lo2 = min(env[i + 1:i + 13])
            if e - max(lo, lo2) > 3:
                syll += 1; last = i
    pauses, run, seen = [], 0, False
    for fr in frames:
        if fr["speech"]:
            if seen and run * hop / fs >= 0.25:
                pauses.append(run * hop / fs)
            run, seen = 0, True
        elif seen:
            run += 1

    return {
        "ok": True, "durationS": dur, "speechS": speech_s,
        "f0MedianHz": _q(f0s, 0.5) if f0s else None, "f0MeanHz": sum(f0s) / len(f0s) if f0s else None,
        "f0RangeSt": 12 * math.log2(_q(f0s, 0.9) / _q(f0s, 0.1)) if len(f0s) > 5 else None,
        "hnrDb": 10 * math.log10(r_mean / (1 - r_mean)) if r_mean else None,
        "formants": None,  # needs LPC analysis; planned for the ROCm build
        "spectralCentroidHz": centroid, "spectralSlopeDbOct": slope,
        "lowMidRelDb": rel(150, 500), "presenceRelDb": rel(2000, 5000), "sibilanceRelDb": rel(5000, 9000),
        "sibilancePeakDb": _q(sib_lv, 0.95) if sib_lv else None, "sibilanceMedianDb": _q(sib_lv, 0.5) if sib_lv else None,
        "rmsDb": rms_db, "integratedLufs": integ, "loudnessRangeLu": lra, "crestDb": lin_to_db(pk) - rms_db,
        "noiseFloorDb": noise_floor, "syllablesPerSec": syll / speech_s if speech_s else None,
        "pausesPerMin": len(pauses) / (dur / 60), "pauseMeanS": sum(pauses) / len(pauses) if pauses else 0.0,
    }


def _c(v, lo, hi):
    return min(hi, max(lo, v))


def _r(x, step=1.0):
    return math.floor(x / step + 0.5) * step


def match_to_reference(a: dict, ref: dict, base: dict) -> dict:
    """Personalised settings that move THIS voice toward the reference. Never changes pitch."""
    if not a or not a.get("ok"):
        return {}
    p: dict = {}
    low_gap = ref["lowMidRelDb"] - a["lowMidRelDb"]
    if low_gap > 0:
        p["warmthDb"], p["mudDb"] = _c(_r(low_gap * 0.6 * 2) / 2, 0, 6), 0
    else:
        p["warmthDb"], p["mudDb"] = 0, _c(_r(low_gap * 0.6 * 2) / 2, -6, 0)
    p["presenceDb"] = _c(_r((ref["presenceRelDb"] - a["presenceRelDb"]) * 0.6 * 2) / 2, -3, 6)
    sib_ex = a["sibilanceRelDb"] - ref["sibilanceRelDb"]
    p["deEssEnabled"] = True
    p["deEssMaxDb"] = _c(_r(4 + max(0, sib_ex) * 1.2), 3, 12)
    in_gain = base.get("inputGainDb", 0)
    if a.get("sibilancePeakDb") is not None:
        p["deEssThresholdDb"] = _c(_r(a["sibilanceMedianDb"] + in_gain + 6 - max(0, sib_ex)), -60, -6)
    lra = a["loudnessRangeLu"] if a.get("loudnessRangeLu") is not None else ref["loudnessRangeLu"]
    p["compEnabled"] = True
    p["compRatio"] = _c(_r((2 + max(0, lra - ref["loudnessRangeLu"]) * 0.35) * 10) / 10, 1.5, 6)
    p["compThresholdDb"] = _c(_r(a["rmsDb"] + in_gain - 2), -50, -6)
    p["makeupDb"] = _c(_r((1 - 1 / p["compRatio"]) * 6 * 2) / 2, 0, 8)
    if a["rmsDb"] < -40:
        p["inputGainDb"] = _c(_r(-30 - a["rmsDb"]), 0, 18)
    p["nsEnabled"] = True
    nf = a.get("noiseFloorDb")
    p["nsAmount"] = 0.85 if (nf is not None and nf > -45) else 0.65 if (nf is not None and nf > ref["noiseFloorDb"]) else 0.4
    p["loudEnabled"], p["targetLufs"] = True, _c(ref["integratedLufs"], -30, -10)
    p["limiterEnabled"], p["limiterCeilingDb"] = True, _c(ref["truePeakDb"], -6, 0)
    return p
