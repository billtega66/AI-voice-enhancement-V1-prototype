"""Stage-level checks of the server engine (same acceptance criteria as the browser self-test)."""
import numpy as np
import pytest

from conftest import level, tone
from voice_engine.dsp import NoiseSuppressor, Pipeline, SpeechGate, integrated_lufs, make_sample_voice, render_offline
from voice_engine.profile import DEFAULTS, SCHEMA

FS = 48000
ALL_OFF = {**DEFAULTS, "nsEnabled": False, "vadEnabled": False, "hpfEnabled": False, "deEssEnabled": False,
           "compEnabled": False, "loudEnabled": False, "limiterEnabled": False, "makeupDb": 0}


def only(**kw):
    return {**ALL_OFF, **kw}


def render(x, p, chunk=1024):
    return render_offline(x, FS, p, chunk)[0]


def test_noise_suppression_reduces_noise_keeps_speech():
    s = make_sample_voice(FS, 4, events=[{"type": "speech", "s": 1.2, "e": 3.2, "db": -22}], noise_db=-40)
    o = render(s.data, only(nsEnabled=True, nsAmount=0.6))
    assert level(o, 3.5, 3.95) - level(s.data, 3.5, 3.95) < -9
    assert level(o, 1.6, 3.0) - level(s.data, 1.6, 3.0) > -3


def test_noise_suppressor_latency_is_exact():
    ns = NoiseSuppressor(FS); ns.amount = 0
    x = np.zeros(4096, np.float32); x[100] = 1
    ns.process(x)
    assert int(np.argmax(np.abs(x))) - 100 == ns.latency
    assert abs(np.max(np.abs(x)) - 1) < 1e-3


def test_vad_detects_speech_but_not_keyboard_or_paper(sample):
    g = SpeechGate(FS)
    hop, tally = g.hop, {"speech": [0, 0], "keys": [0, 0], "paper": [0, 0]}
    for i in range(0, len(sample.data) - hop + 1, hop):
        b = sample.data[i:i + hop].copy(); g.process(b, False)
        t = i / FS; lab = sample.label_at(t)
        ev = next((e for e in sample.events if e["s"] <= t < e["e"]), None)
        if lab == "speech" and (t - ev["s"] < 0.15 or ev["e"] - t < 0.1):
            continue
        if lab in tally:
            tally[lab][0] += 1; tally[lab][1] += g.speech
    assert tally["speech"][1] / tally["speech"][0] > 0.9
    assert tally["keys"][1] / tally["keys"][0] < 0.05
    assert tally["paper"][1] / tally["paper"][0] < 0.05


@pytest.mark.parametrize("key,freq,gain", [("warmthDb", 180, 6), ("presenceDb", 4000, 6), ("mudDb", 350, -6), ("airDb", 14000, 6)])
def test_eq_bands(key, freq, gain):
    x = tone(freq, 0.6)
    assert abs(level(render(x, only(**{key: gain})), 0.3, 0.6) - level(x, 0.3, 0.6) - gain) < 1.2


def test_deesser_reduces_only_sibilant_band():
    p = only(deEssEnabled=True, deEssFreqHz=6500, deEssThresholdDb=-30, deEssMaxDb=10)
    hi, lo = tone(6500, 0.6, -10), tone(500, 0.6, -10)
    assert level(render(hi, p), 0.3, 0.6) + 10 < -5
    assert abs(level(render(lo, p), 0.3, 0.6) - level(lo, 0.3, 0.6)) < 0.5


def test_compressor_narrows_dynamic_range():
    x = np.concatenate([tone(300, 1, -36), tone(300, 1, -12)])
    o = render(x, only(compEnabled=True, compThresholdDb=-36, compRatio=4, compAttackMs=5, compReleaseMs=100, makeupDb=0))
    assert level(o, 1.5, 2) - level(o, 0.5, 1) < 9


def test_loudness_normalisation_hits_target():
    s = make_sample_voice(FS, 10, events=[{"type": "speech", "s": 0.3, "e": 9.7, "db": -40}])
    o = render(s.data, only(loudEnabled=True, targetLufs=-16))
    assert abs(integrated_lufs(o[6 * FS:], FS)[0] + 16) < 1.5


def test_limiter_ceiling_and_output_gain():
    x = tone(440, 1, 0); x[::997] = 2.5
    o = render(x, only(limiterEnabled=True, limiterCeilingDb=-1, outputGainDb=6))
    assert 20 * np.log10(np.max(np.abs(o))) <= -1 + 1e-4
    y = tone(440, 1, -30)
    assert abs(level(render(y, only(outputGainDb=-6)), 0.2, 1) - level(y, 0.2, 1) + 6) < 0.05


def test_bypass_is_bit_exact(sample):
    pl = Pipeline(FS, {**DEFAULTS, "warmthDb": 4, "pitchSemitones": 1})
    for i in range(0, len(sample.data) - 1024 + 1, 1024):
        b = sample.data[i:i + 1024].copy()
        pl.process(b, bypass=True)
        assert np.array_equal(b, sample.data[i:i + 1024])


def test_output_independent_of_chunk_size():
    s = make_sample_voice(FS, 4)
    a, b = render(s.data, DEFAULTS, 256), render(s.data, DEFAULTS, 2048)
    assert np.max(np.abs(a - b)) < 1e-6


def test_every_profile_parameter_changes_the_audio():
    s = make_sample_voice(FS, 6, events=[{"type": "speech", "s": 0.5, "e": 2.5, "db": -22}, {"type": "keys", "s": 2.8, "e": 3.4, "db": -24},
                                         {"type": "speech", "s": 3.8, "e": 5.6, "db": -46}])
    base = {**DEFAULTS, "outputGainDb": 4, "limiterCeilingDb": -3, "compThresholdDb": -40, 'metallicMix': 0.2, 'distortionDrive': 0.2, 'echoMix': 0.2}
    ref = render(s.data, base, 512)
    dead = []
    for k, sc in SCHEMA.items():
        v = (not base[k]) if sc["type"] == "bool" else (sc["max"] if sc["max"] - base[k] > base[k] - sc["min"] else sc["min"])
        if not np.mean(np.abs(render(s.data, {**base, k: v}, 512) - ref)) > 1e-6:
            dead.append(k)
    assert dead == []


def test_stable_at_extreme_settings():
    p = {**DEFAULTS, **{k: s["max"] for k, s in SCHEMA.items() if s["type"] == "num"}}
    pl = Pipeline(FS, p)
    rng = np.random.default_rng(9)
    for k in range(300):
        b = np.zeros(512, np.float32) if k % 3 == 0 else (rng.uniform(-5, 5, 512).astype(np.float32) if k % 3 == 1 else np.full(512, 1e-30, np.float32))
        pl.process(b)
        assert np.all(np.isfinite(b)) and np.max(np.abs(b)) <= 1


def test_faster_than_real_time(sample):
    import time
    render(sample.data[:FS], DEFAULTS)  # JIT warm-up
    t0 = time.perf_counter(); render(sample.data, DEFAULTS)
    assert (time.perf_counter() - t0) / (len(sample.data) / FS) < 0.5
