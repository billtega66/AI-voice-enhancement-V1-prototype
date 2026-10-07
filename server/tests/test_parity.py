"""The browser engine (web/src/*.js) and the server engine must agree, so a profile sounds
the same whichever engine runs it, and the AI/Mixer contract is identical on both sides."""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import NODE, ROOT, needs_node, run_node
from voice_engine.ai import local_interpret
from voice_engine.analysis import analyze_voice, match_to_reference
from voice_engine.dsp import render_offline
from voice_engine.profile import DEFAULT_REFERENCE, DEFAULTS, SCHEMA, validate_changes

PROFILES = [
    {},
    {"nsEnabled": False, "vadEnabled": False, "hpfEnabled": False, "deEssEnabled": False, "compEnabled": False, "loudEnabled": False, "limiterEnabled": False, "makeupDb": 0},
    {"pitchSemitones": -1.5, "warmthDb": 4, "presenceDb": 3, "airDb": 2, "mudDb": -3, "inputGainDb": 6},
    {"compRatio": 6, "compThresholdDb": -40, "deEssThresholdDb": -50, "deEssMaxDb": 12, "nsAmount": 1, "vadSensitivity": 1, "targetLufs": -12, "limiterCeilingDb": -3},
]


@pytest.fixture(scope="module")
def js_renders():
    d = Path(tempfile.mkdtemp())
    subprocess.run([NODE, str(ROOT / "tests/js/render_fixture.js"), str(d), json.dumps(PROFILES), "1024"], check=True)
    return d


@needs_node
@pytest.mark.parametrize("i", range(len(PROFILES)))
def test_audio_matches_browser_engine(js_renders, i):
    x = np.fromfile(js_renders / "input.f32", np.float32)
    js = np.fromfile(js_renders / f"out{i}.f32", np.float32)
    py, infos, _ = render_offline(x, 48000, {**DEFAULTS, **PROFILES[i]})
    assert js.shape == py.shape
    assert np.max(np.abs(js - py)) < 1e-5, "server output drifted from the browser engine"
    jsinf = json.loads((js_renders / f"info{i}.json").read_text())
    assert [a["speech"] for a in jsinf] == [b.speech for b in infos]


@needs_node
def test_schema_is_shared():
    js = json.loads(run_node("process.stdout.write(JSON.stringify(VoiceProfile.SCHEMA))"))
    assert js == SCHEMA


CASES = [{"warmthDb": 50, "presenceDb": "2.3", "compEnabled": "false", "hackerKey": 1, "airDb": "loud"},
         {"compRatio": 3.333, "vadSensitivity": -1, "deEssEnabled": 1, "nsAmount": 0.123}]


@needs_node
@pytest.mark.parametrize("case", CASES)
def test_validation_matches(case):
    js = json.loads(run_node(f"process.stdout.write(JSON.stringify(VoiceProfile.validateChanges({json.dumps(case)})))"))
    ok, rej = validate_changes(case)
    assert js["ok"] == ok and sorted(js["rejected"]) == sorted(rej)


PROMPTS = [
    "Make my voice warmer, clearer, and more professional, like a podcast.",
    "My voice sounds thin and inconsistent. Make it warmer and keep the volume consistent.",
    "Good, but make it slightly deeper and reduce the sharp S sounds.",
    "Keep everything else but make it slightly clearer.",
    "there is keyboard and air conditioner noise in the background",
    "a bit less bright, much louder",
    "make it more natural",
    "start over",
    "words get cut off",
    "asdf qwerty",
]


@needs_node
@pytest.mark.parametrize("prompt", PROMPTS)
def test_offline_interpreter_matches(prompt):
    prof = {**DEFAULTS, "warmthDb": 1.5, "compRatio": 3}
    js = json.loads(run_node(f"""const r = VoiceAI.localInterpret({json.dumps(prompt)}, {json.dumps(prof)}, null, VoiceAnalysis.DEFAULT_REFERENCE);
      process.stdout.write(JSON.stringify({{ok: VoiceProfile.validateChanges(r.changes).ok, matched: r.matched}}))"""))
    py = local_interpret(prompt, prof, None, DEFAULT_REFERENCE)
    assert js["matched"] == py["matched"]
    assert js["ok"] == validate_changes(py["changes"])[0]


@needs_node
def test_reference_matching_matches(sample):
    a = analyze_voice(sample.data, 48000)
    js = json.loads(run_node(f"process.stdout.write(JSON.stringify(VoiceAnalysis.matchToReference({json.dumps(a)}, VoiceAnalysis.DEFAULT_REFERENCE, VoiceProfile.DEFAULTS)))"))
    assert js == match_to_reference(a, DEFAULT_REFERENCE, DEFAULTS)


@needs_node
def test_analysis_agrees_with_browser():
    """Same input through both analysers: the measurements should agree closely."""
    d = Path(tempfile.mkdtemp())
    subprocess.run([NODE, str(ROOT / "tests/js/render_fixture.js"), str(d), "[]"], check=True)
    x = np.fromfile(d / "input.f32", np.float32)
    js = json.loads(run_node(f"""const fs=require('fs'); const b=fs.readFileSync({json.dumps(str(d / 'input.f32'))});
      const x=new Float32Array(b.buffer,b.byteOffset,b.length/4); process.stdout.write(JSON.stringify(VoiceAnalysis.analyzeVoice(x,48000)))"""))
    py = analyze_voice(x, 48000)
    for k, tol in [("f0MedianHz", 2), ("integratedLufs", 0.3), ("lowMidRelDb", 0.5), ("presenceRelDb", 0.5), ("sibilanceRelDb", 0.5),
                   ("rmsDb", 0.3), ("noiseFloorDb", 1), ("pausesPerMin", 0.5), ("speechS", 0.05)]:
        assert abs(js[k] - py[k]) <= tol, f"{k}: browser {js[k]} vs server {py[k]}"
