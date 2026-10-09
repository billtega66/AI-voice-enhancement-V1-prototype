"""Three real AI planning checks. Uses API quota; no offline fallback."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from voice_engine.cli import load_env_file
from voice_engine.ai.gateway import GatewayInterpreter, CallLedger
from voice_engine.profile import DEFAULTS, DEFAULT_REFERENCE

load_env_file(ROOT / '.env')
gateway = GatewayInterpreter(CallLedger(':memory:'))
cases = [
    ('Make my voice metallic and robotic with a short repeating echo.', 'modules'),
    ('Clone another speaker exactly, including their accent and speaking rhythm.', 'unsupported'),
    ('Make my voice different.', 'clarify'),
]
results = []
for index, (request, expected) in enumerate(cases):
    if index:
        time.sleep(30)
    row = {'request': request, 'expected': expected}
    try:
        answer = gateway.interpret(request, {'profile': dict(DEFAULTS), 'reference': DEFAULT_REFERENCE,
                                            'analysis': None, 'history': [], 'mode': 'creative'})
        row['answer'] = answer
        if expected == 'modules':
            passed = answer['changes'].get('metallicMix', 0) > 0 and answer['changes'].get('echoMix', 0) > 0
        else:
            passed = not answer['changes'] and (answer['intent'] == expected or answer.get('support') == expected)
        row['status'] = 'PASS' if passed else 'FAIL'
    except Exception as exc:
        row.update(status='ERROR', error=str(exc))
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    path = ROOT / 'reports' / 'modular-voice-live-check.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'model': gateway.model, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
