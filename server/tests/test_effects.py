"""Signal behavior, state continuity, ordering, and browser/CPU equivalence."""
import json
import numpy as np
import pytest
from conftest import needs_node, run_node
from voice_engine.dsp.effects import EffectsChain
from voice_engine.dsp.engine import Pipeline
from voice_engine.profile import DEFAULTS

FS = 8000


def signal():
    return (0.2 * np.sin(2 * np.pi * 440 * np.arange(4000) / FS)).astype(np.float32)


def process(data, settings, chunk=512):
    out = data.copy()
    chain = EffectsChain(FS)
    p = {**DEFAULTS, **settings}
    for start in range(0, len(out), chunk):
        chain.process(out[start:start + chunk], p)
    return out


def test_disabled_modules_preserve_audio():
    data = signal()
    np.testing.assert_array_equal(process(data, {}), data)


def test_metallic_module_creates_sidebands():
    out = process(signal(), {'metallicMix': 1, 'metallicHz': 60})
    spectrum = np.abs(np.fft.rfft(out))
    assert spectrum[190] > 50 and spectrum[250] > 50  # 380 and 500 Hz
    assert spectrum[220] < 0.01  # carrier at 440 Hz removed


def test_echo_has_expected_delay_and_feedback():
    impulse = np.zeros(2000, dtype=np.float32); impulse[0] = .4
    out = process(impulse, {'echoMix': .5, 'echoMs': 40, 'echoFeedback': .5})
    assert out[0] == pytest.approx(.2)
    assert out[320] == pytest.approx(.2)
    assert out[640] == pytest.approx(.1)


def test_distortion_changes_signal_without_nonfinite_values():
    data = signal()
    out = process(data, {'distortionDrive': .5})
    assert np.isfinite(out).all() and np.max(np.abs(out)) < 1
    assert not np.allclose(data, out)


def test_order_changes_audio_and_chunks_preserve_state():
    p = {'metallicMix': .6, 'echoMix': .4, 'echoMs': 70, 'distortionDrive': .3}
    out = process(signal(), p, 128)
    np.testing.assert_array_equal(out, process(signal(), p, 1024))
    assert not np.allclose(out, process(signal(), {**p, 'echoBeforeTexture': True}))


def test_pipeline_runs_modules_before_final_limiter():
    p = {**DEFAULTS, 'nsEnabled': False, 'vadEnabled': False, 'hpfEnabled': False,
         'deEssEnabled': False, 'compEnabled': False, 'loudEnabled': False,
         'metallicMix': .4, 'echoMix': .4, 'echoMs': 40, 'outputGainDb': 6}
    audio = signal()[:512]
    Pipeline(FS, p).process(audio)
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) <= 10 ** (p['limiterCeilingDb'] / 20) + 1e-6
    assert not np.allclose(audio, signal()[:512])


@needs_node
@pytest.mark.parametrize('settings', [
    {'metallicMix': .7, 'metallicHz': 83.5}, {'distortionDrive': .4},
    {'echoMix': .4, 'echoMs': 70, 'echoFeedback': .6},
    {'metallicMix': .4, 'distortionDrive': .3, 'echoMix': .4, 'echoBeforeTexture': True},
])
def test_browser_cpu_effect_parity(settings):
    data = signal()
    js = run_node("const data=Float32Array.from({length:4000},(_,i)=>0.2*Math.sin(2*Math.PI*440*i/8000));"
                  f"const p={{...VoiceProfile.DEFAULTS,...{json.dumps(settings)}}};"
                  "const chain=new DSP.EffectsChain(8000);"
                  "for(let i=0;i<data.length;i+=333)chain.process(data.subarray(i,i+333),p);"
                  "console.log(JSON.stringify(Array.from(data)));")
    np.testing.assert_allclose(process(data, settings), np.asarray(json.loads(js)), atol=1e-7, rtol=0)
