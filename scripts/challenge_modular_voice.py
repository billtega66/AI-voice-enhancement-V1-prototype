"""Run challenging real-model requests. Uses API quota; no offline fallback.

Checks are deliberately coarse. Read the replies and listen to candidate audio
to evaluate whether the claimed sound is delivered. Never treat legal values as
proof of a convincing effect.
"""
import argparse
import json
import sys
import time
import httpx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from voice_engine.cli import load_env_file
from voice_engine.ai.gateway import GatewayInterpreter, CallLedger
from voice_engine.profile import DEFAULTS, DEFAULT_REFERENCE


def assess(case, answer, profile):
    changes = answer['changes']
    reasons = []
    for key in case.get('positive', []):
        if changes.get(key, profile[key]) <= profile[key]:
            reasons.append(key + ' did not increase')
    for key in case.get('unchanged', []):
        if changes.get(key, profile[key]) != profile[key]:
            reasons.append(key + ' changed despite the constraint')
    for key in case.get('zero', []):
        if changes.get(key, profile[key]) != 0:
            reasons.append(key + ' was not removed')
    if case['expect'] == 'clarify' and (changes or answer['intent'] != 'clarify'):
        reasons.append('Expected clarification without changing audio')
    if case['expect'] == 'limited':
        if answer.get('support') == 'supported' and answer['intent'] not in ('clarify', 'unsupported'):
            reasons.append('Did not acknowledge the missing processing capability')
        if answer.get('support') == 'unsupported' and changes:
            reasons.append('Unsupported request changed audio')
    if case['expect'] == 'enhancement' and changes.get('deEssThresholdDb', profile['deEssThresholdDb']) >= profile['deEssThresholdDb'] and changes.get('deEssMaxDb', profile['deEssMaxDb']) <= profile['deEssMaxDb']:
        reasons.append('S-sound control did not strengthen')
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=35)
    parser.add_argument('--only', default='', help='Comma-separated case IDs for a follow-up check')
    parser.add_argument('--output', default='', help='Report path relative to the repository')
    args = parser.parse_args()
    load_env_file(ROOT / '.env')
    diagnostics = {}
    def capture_error(response):
        diagnostics.clear()
        if response.status_code >= 400:
            response.read()
            try:
                error = response.json().get('error', {})
                diagnostics.update(code=error.get('code'), message=str(error.get('message', ''))[:1000])
            except (ValueError, AttributeError):
                diagnostics['code'] = 'unreadable_provider_error'
    client = httpx.Client(follow_redirects=False, event_hooks={'response': [capture_error]})
    gateway = GatewayInterpreter(CallLedger(':memory:'), client=client)
    cases = json.loads((ROOT / 'tests/evaluations/modular_voice_challenges.json').read_text(encoding='utf-8'))
    if args.only:
        cases = [case for case in cases if case['id'] in args.only.split(',')]
    path = ROOT / ('reports/modular-voice-challenges-recheck.json' if args.only else 'reports/modular-voice-challenges.json')
    if args.output:
        path = ROOT / args.output
    path.parent.mkdir(exist_ok=True)
    results = []
    for index, case in enumerate(cases):
        if index:
            time.sleep(max(0, args.interval))
        profile = {**DEFAULTS, **case.get('profile', {})}
        row = {'id': case['id'], 'mode': case['mode'], 'request': case['request'], 'expect': case['expect']}
        started = time.monotonic()
        try:
            answer = gateway.interpret(case['request'], {'profile': profile, 'reference': DEFAULT_REFERENCE,
                'analysis': None, 'history': [], 'mode': case['mode']})
            row['answer'] = answer
            row['findings'] = assess(case, answer, profile)
            row['status'] = 'FAIL' if row['findings'] else 'PASS'
        except Exception as exc:
            row.update(status='ERROR', error=str(exc))
            if diagnostics:
                row['provider_error'] = dict(diagnostics)
        row['seconds'] = round(time.monotonic() - started, 2)
        results.append(row)
        path.write_text(json.dumps({'model': gateway.model, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    lines = ['# Modular voice challenges', '', '| Request | Result | AI reply | Changes |', '|---|---|---|---|']
    for row in results:
        answer = row.get('answer', {})
        cells = [row['request'], row['status'], answer.get('reply', row.get('error', '')), json.dumps(answer.get('changes', {}))]
        lines.append('| ' + ' | '.join(str(cell).replace('|', '\\|').replace('\n', ' ') for cell in cells) + ' |')
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    client.close()


if __name__ == '__main__':
    main()
