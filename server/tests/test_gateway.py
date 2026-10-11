"""OpenAI-compatible gateway (Bailian DeepSeek / vLLM on ROCm): transport, validation, budget, fallback.
All model replies are mocked with httpx.MockTransport; no network, no cost."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from voice_engine.ai import (BudgetExceeded, CallLedger, GatewayInterpreter, ModelUnavailable, build_prompts, call_model,
                             check_answer, get_interpreter)
from voice_engine.profile import DEFAULT_REFERENCE, DEFAULTS
from voice_engine.server import create_app


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://gw.example/v1/")
    monkeypatch.setenv("LLM_MODEL", "bailian/deepseek-v4.1-flash")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_THINKING_PARAM", raising=False)
    monkeypatch.setenv("LLM_DAILY_CALL_LIMIT", "200")
    monkeypatch.setenv("LLM_PER_CLIENT_PER_MIN", "8")


def reply(content, finish="stop", status=200, seen=None):
    def handler(req: httpx.Request):
        if seen is not None:
            seen.append(req)
        return httpx.Response(status, json={"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def ctx(**kw):
    return {"profile": dict(DEFAULTS), "analysis": None, "reference": DEFAULT_REFERENCE, "history": [], "owner": "t", **kw}


def interp(client, ledger=None):
    return GatewayInterpreter(ledger or CallLedger(":memory:"), client)


def test_request_shape_bailian():
    seen = []
    r = interp(reply('{"reply":"Warmer.","changes":{"warmthDb":2}}', seen=seen)).interpret("make it warmer", ctx())
    assert r["changes"] == {"warmthDb": 2}
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://gw.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "bailian/deepseek-v4.1-flash" and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"} and body["enable_thinking"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    user = json.loads(body["messages"][1]["content"])
    assert user["request"] == "make it warmer"  # untrusted text travels as data
    assert "UNTRUSTED DATA" in body["messages"][0]["content"]


def test_request_shape_vllm(monkeypatch):
    monkeypatch.setenv("LLM_THINKING_PARAM", "chat_template_kwargs")
    seen = []
    interp(reply('{"reply":"ok","changes":{}}', seen=seen)).interpret("hi", ctx())
    body = json.loads(seen[0].content)
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and "enable_thinking" not in body


def test_request_shape_openai(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_THINKING_PARAM", "none")
    seen = []
    interp(reply('{"reply":"Warmer.","changes":{"warmthDb":2}}', seen=seen)).interpret("make it warmer", ctx())
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.openai.com/v1/chat/completions"
    assert body["model"] == "gpt-4.1-mini"
    assert body["response_format"] == {"type": "json_object"}
    assert "enable_thinking" not in body and "chat_template_kwargs" not in body


@pytest.mark.parametrize("content,finish,status,msg", [
    ('{"reply":"x","changes":{}}', "length", 200, "incomplete"),
    ('{"reply":"x","changes":{}}', "stop", 500, "HTTP 500"),
    ("not json", "stop", 200, "failed"),
    ('["a"]', "stop", 200, "not a JSON object"),
])
def test_transport_failures(content, finish, status, msg):
    with pytest.raises(ModelUnavailable, match=msg):
        call_model("s", "u", client=reply(content, finish, status))


def test_timeout_and_redirect():
    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)
    with pytest.raises(ModelUnavailable, match="too long"):
        call_model("s", "u", client=httpx.Client(transport=httpx.MockTransport(slow)))
    redir = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(302, headers={"location": "http://evil/"})))
    with pytest.raises(ModelUnavailable, match="redirected"):
        call_model("s", "u", client=redir)


def test_fenced_json_is_accepted():
    assert call_model("s", "u", client=reply('```json\n{"reply":"a","changes":{}}\n```')) == {"reply": "a", "changes": {}}


@pytest.mark.parametrize("obj,text,msg", [
    ({"changes": {}}, "warmer", "wrong shape"),
    ({"reply": "x", "changes": {}, "secret": 1}, "warmer", "unexpected fields"),
    ({"reply": "x", "changes": {"hackerKey": 1}}, "warmer", "do not exist"),
])
def test_semantic_validation_rejects(obj, text, msg):
    with pytest.raises(ModelUnavailable, match=msg):
        check_answer(obj, text, dict(DEFAULTS))


def test_semantic_validation_accepts_and_quantises():
    r = check_answer({"reply": "Deeper.", "changes": {"pitchSemitones": -1, "warmthDb": 2.3}}, "slightly deeper", dict(DEFAULTS))
    assert r["changes"] == {"pitchSemitones": -1, "warmthDb": 2.5}


def test_prompt_carries_current_profile_and_history():
    prof = {**DEFAULTS, "warmthDb": 4.5}
    system, user = build_prompts("clearer", ctx(profile=prof, history=[{"role": "user", "content": "podcast"}, {"role": "assistant", "content": "done"}]))
    assert "warmthDb (Warmth (180 Hz)): -6..9 dB, step 0.5, now 4.5" in system
    assert json.loads(user)["recentConversation"] == [{"role": "user", "text": "podcast"}, {"role": "assistant", "text": "done"}]


def test_budget_daily_and_per_client(monkeypatch):
    led = CallLedger(":memory:")
    monkeypatch.setenv("LLM_PER_CLIENT_PER_MIN", "2")
    led.reserve("a", 1000.0); led.reserve("a", 1001.0)
    with pytest.raises(BudgetExceeded, match="wait"):
        led.reserve("a", 1002.0)
    led.reserve("a", 1100.0)  # a minute later
    monkeypatch.setenv("LLM_DAILY_CALL_LIMIT", "4")
    led.reserve("b", 1101.0)
    with pytest.raises(BudgetExceeded, match="today"):
        led.reserve("c", 1102.0)
    led.reserve("c", 86400.0 + 5)  # next UTC day
    assert led.used_today(86400.0 + 6) == 1


def test_failed_attempts_count_toward_budget(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_CALL_LIMIT", "1")
    led = CallLedger(":memory:")
    g = interp(reply("broken"), led)
    with pytest.raises(ModelUnavailable):
        g.interpret("warmer", ctx())
    with pytest.raises(BudgetExceeded):
        g.interpret("warmer", ctx())


def test_selection_order(monkeypatch):
    assert get_interpreter().name == "gateway"
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_interpreter().name == "offline"


def test_api_uses_gateway_then_falls_back(monkeypatch, tmp_path):
    answers = iter(['{"reply":"Clearer now.","changes":{"presenceDb":1}}', '{"reply":"x","changes":{"bogus":1}}'])
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": next(answers)}}]})))
    app = create_app("cpu", GatewayInterpreter(CallLedger(str(tmp_path / "db.sqlite")), client))
    c = TestClient(app)
    assert c.get("/api/health").json()["model"] == "bailian/deepseek-v4.1-flash"
    r = c.post("/api/interpret", json={"text": "slightly clearer", "profile": {"warmthDb": 4.5}})
    j = r.json()
    assert j["interpreter"] == "gateway" and j["changes"] == {"presenceDb": 1} and j["note"] is None
    assert "voice_guest" in r.cookies
    j = c.post("/api/interpret", json={"text": "make it warmer"}).json()
    assert j["interpreter"] == "offline" and "do not exist" in j["note"] and j["changes"]["warmthDb"] == 2


def test_api_rate_limit_per_browser(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PER_CLIENT_PER_MIN", "1")
    ok = '{"reply":"ok","changes":{"warmthDb":1}}'
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": ok}}]})))
    c = TestClient(create_app("cpu", GatewayInterpreter(CallLedger(str(tmp_path / "db.sqlite")), client)))
    assert c.post("/api/interpret", json={"text": "warmer"}).json()["interpreter"] == "gateway"
    j = c.post("/api/interpret", json={"text": "warmer"}).json()
    assert j["interpreter"] == "offline" and "wait" in j["note"]


@pytest.mark.parametrize("text,pitch", [("super deep voice, alpha male", -5), ("dramatic cinematic voice", -7), ("very high voice", 7)])
def test_character_requests_allow_strong_pitch(text, pitch):
    r = check_answer({"reply": "Transformed.", "intent":"transformation", "changes": {"pitchSemitones": pitch}}, text, dict(DEFAULTS))
    assert r["changes"]["pitchSemitones"] == pitch


def test_enhancement_keeps_pitch_subtle_without_discarding_tone():
    r = check_answer({"reply":"x", "intent":"enhancement", "changes":{"pitchSemitones":-7,"presenceDb":2}}, "less dramatic", dict(DEFAULTS))
    assert r["changes"] == {"pitchSemitones":-1.5,"presenceDb":2}
    assert "subtle" in r["note"]



@pytest.mark.parametrize("presence,expected", [(20, 9), (-20, -6), ("20", 9)])
def test_numeric_limits_keep_other_voice_changes(presence, expected):
    r = check_answer({"reply": "Dramatic.", "intent":"transformation", "changes": {"presenceDb": presence, "pitchSemitones": -7}}, "dramatic voice", dict(DEFAULTS))
    assert r["changes"] == {"presenceDb": expected, "pitchSemitones": -7}
    assert "Presence" in r["note"] and "limited" in r["note"]


def test_api_reports_limited_settings_without_offline_fallback(tmp_path):
    client = reply('{"reply":"Dramatic.","intent":"transformation","changes":{"presenceDb":20,"pitchSemitones":-7}}')
    c = TestClient(create_app("cpu", GatewayInterpreter(CallLedger(str(tmp_path / "limits.sqlite")), client)))
    j = c.post("/api/interpret", json={"text":"dramatic voice"}).json()
    assert j["interpreter"] == "gateway"
    assert j["changes"] == {"presenceDb":9, "pitchSemitones":-7}
    assert "limited" in j["note"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "loud"])
def test_invalid_numeric_values_still_reject(value):
    with pytest.raises(ModelUnavailable):
        check_answer({"reply":"x", "changes":{"presenceDb":value}}, "clearer", dict(DEFAULTS))


@pytest.mark.parametrize("text", ["girl-like-voice", "female voice", "feminine voice", "woman voice"])
def test_feminine_request_accepts_pitch_transformation(text):
    r = check_answer({"reply":"Brighter and higher.","intent":"transformation","changes":{"pitchSemitones":5,"warmthDb":-2}}, text, dict(DEFAULTS))
    assert r["changes"]["pitchSemitones"] == 5


@pytest.mark.parametrize("text", ["a tiny woodland sprite", "a gravelly giant telling a story", "giọng nhẹ và thanh", "less of that, keep the character"])
def test_freeform_transformations_do_not_require_pitch_keywords(text):
    r = check_answer({"reply":"Applied.", "intent":"transformation", "changes":{"pitchSemitones":4.25,"presenceDb":1.5}}, text, dict(DEFAULTS))
    assert r["changes"]["pitchSemitones"] == 4.25
    assert r["intent"] == "transformation"


def test_invalid_intent_rejects():
    with pytest.raises(ModelUnavailable, match="wrong shape"):
        check_answer({"reply":"x", "intent":"invalid", "changes":{}}, "x", dict(DEFAULTS))
