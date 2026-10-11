import json
import types

import numpy as np
import soundfile as sf

from voice_engine.ai import ClaudeInterpreter, build_messages, parse_ai_response
from voice_engine.cli import main
from voice_engine.profile import DEFAULT_REFERENCE, DEFAULTS


def test_cli_enhance_analyze_benchmark_interpret(tmp_path, sample, capsys):
    src = tmp_path / "in.wav"; dst = tmp_path / "out.wav"; prof = tmp_path / "p.json"
    sf.write(src, sample.data[:48000 * 3], 48000, subtype="FLOAT")
    prof.write_text(json.dumps({"params": {"warmthDb": 3}}))
    assert main(["enhance", str(src), str(dst), "--profile", str(prof), "--prompt", "less background noise"]) == 0
    out, fs = sf.read(dst, dtype="float32")
    assert fs == 48000 and out.shape[0] == 48000 * 3 and np.max(np.abs(out)) <= 1
    assert main(["analyze", str(src)]) == 0 and '"ok": true' in capsys.readouterr().out
    csv = tmp_path / "b.csv"
    assert main(["benchmark", "--chunks", "1024", "--seconds", "2", "--csv", str(csv)]) == 0
    assert csv.read_text().startswith("backend,chunk")
    assert main(["interpret", "make it warmer"]) == 0 and '"warmthDb": 2' in capsys.readouterr().out
    assert main(["backends"]) == 0


def test_parse_ai_response_variants():
    r = parse_ai_response('Sure!\n```json\n{"reply":"ok","changes":{"warmthDb":50,"bad":1}}\n```')
    assert r["changes"] == {"warmthDb": 9} and r["rejected"] == ["bad"] and r["reply"] == "ok"
    assert parse_ai_response("no json here")["rejected"] == ["response was not a JSON object"]


def test_claude_interpreter_sends_current_profile_and_history():
    calls = []

    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text='{"reply":"Warmer now.","changes":{"warmthDb":3}}')])

    ci = ClaudeInterpreter(client=types.SimpleNamespace(messages=Msgs()), model="test-model")
    prof = {**DEFAULTS, "compRatio": 4}
    hist = [{"role": "user", "content": "podcast voice"}, {"role": "assistant", "content": '{"reply":"done"}'}]
    res = ci.interpret("slightly warmer", {"profile": prof, "analysis": None, "reference": DEFAULT_REFERENCE, "history": hist})
    assert res["changes"] == {"warmthDb": 3} and res["reply"] == "Warmer now."
    kw = calls[0]
    assert kw["model"] == "test-model" and "compRatio (Ratio): 1..10 :1, step 0.1, now 4" in kw["system"]
    roles = [m["role"] for m in kw["messages"]]
    assert roles == ["user", "assistant", "user"] and kw["messages"][-1]["content"] == "slightly warmer"


def test_build_messages_alternates_roles():
    _, msgs = build_messages([{"role": "assistant", "content": "x"}, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}], "c",
                             {"profile": DEFAULTS, "reference": DEFAULT_REFERENCE})
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user" and all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def test_character_presets_are_strong_and_repeatable():
    from voice_engine.ai.interpreter import local_interpret
    from voice_engine.profile import DEFAULTS, DEFAULT_REFERENCE
    for prompt, pitch in [("super deep voice, alpha male", -5), ("deep male voice", -5), ("dramatic cinematic voice", -7)]:
        first = local_interpret(prompt, dict(DEFAULTS), None, DEFAULT_REFERENCE)["changes"]
        assert first["pitchSemitones"] == pitch
        assert first["limiterEnabled"]
        second = local_interpret(prompt, {**DEFAULTS, **first}, None, DEFAULT_REFERENCE)["changes"]
        assert first == second
    assert "pitchSemitones" not in local_interpret("not a dramatic voice", dict(DEFAULTS), None, DEFAULT_REFERENCE)["changes"]



def test_girl_like_voice_maps_to_feminine_preset():
    from voice_engine.ai.interpreter import local_interpret
    from voice_engine.profile import DEFAULTS, DEFAULT_REFERENCE
    r = local_interpret("girl-like-voice", dict(DEFAULTS), None, DEFAULT_REFERENCE)
    assert r["matched"] and r["changes"]["pitchSemitones"] == 5
    assert r["changes"]["warmthDb"] == -2
    assert "pitchSemitones" not in local_interpret("not a feminine voice", dict(DEFAULTS), None, DEFAULT_REFERENCE)["changes"]
