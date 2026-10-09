"""Run real configured AI requests, preserving raw replies and validation results.

Uses API quota. No offline fallback and no automatic retries, so service failures
cannot be mistaken for successful AI answers. Run with Python -X utf8.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from voice_engine.cli import load_env_file
from voice_engine.ai.gateway import build_prompts, check_answer
from voice_engine.profile import DEFAULTS, DEFAULT_REFERENCE
import httpx

CASES = [
    ('enhancement', 'Make my voice warmer.', 'warmthDb', 'up'),
    ('enhancement', 'Reduce the sharp S sounds.', 'deEssThresholdDb', 'down'),
    ('enhancement', 'Remove keyboard noise while keeping my speech clear.', 'nsAmount', 'up'),
    ('enhancement', 'Keep my volume steady without changing my pitch.', 'compRatio', 'up'),
    ('enhancement', 'Make it less bright, but keep the warmth.', 'airDb', 'down'),
    ('enhancement', 'Make my voice slightly deeper.', 'pitchSemitones', 'down'),
    ('creative', 'I want my voice to sound like chipmunks.', 'pitchSemitones', 'up'),
    ('enhancement', 'I want my voice to sound like chipmunks.', None, 'no_changes'),
    ('creative', 'Make me sound like a robot.', None, 'question_or_unsupported'),
    ('creative', 'Make my voice different.', None, 'question_or_unsupported'),
]


def write_markdown(path, metadata, results):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['# Voice AI evaluation', '', 'Model: `' + metadata['model'] + '`',
             '', 'Each request starts from the default profile; no offline fallback is used.',
             'PASS checks the expected parameter direction or no-change behavior, not audio quality.', '',
             '| # | Mode | Request | Result | AI reply / error | Changes |',
             '|---|---|---|---|---|---|']
    for result in results:
        answer = result.get('answer', {})
        values = [result['id'], result['mode'], result['request'], result['status'],
                  answer.get('reply', result.get('error', '')), json.dumps(answer.get('changes', {}), ensure_ascii=False)]
        lines.append('| ' + ' | '.join(cell(value) for value in values) + ' |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def assess(answer, case):
    _, text, key, expectation = case
    changes = answer['changes']
    if expectation == 'no_changes':
        return not changes
    if expectation == 'question_or_unsupported':
        return not changes and answer['intent'] in ('clarify', 'unsupported')
    value = changes.get(key, DEFAULTS[key])
    ok = value > DEFAULTS[key] if expectation == 'up' else value < DEFAULTS[key]
    if 'without changing my pitch' in text:
        ok = ok and changes.get('pitchSemitones', DEFAULTS['pitchSemitones']) == DEFAULTS['pitchSemitones']
    if 'keep the warmth' in text:
        ok = ok and changes.get('warmthDb', DEFAULTS['warmthDb']) == DEFAULTS['warmthDb']
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=25)
    args = parser.parse_args()
    load_env_file(ROOT / '.env')
    output = ROOT / 'reports' / 'voice-ai-evaluation.json'
    output.parent.mkdir(exist_ok=True)
    results = []
    metadata = {'model': os.environ['LLM_MODEL'], 'endpoint': os.environ['LLM_BASE_URL'],
                'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with httpx.Client(follow_redirects=False, timeout=40) as client:
        for index, case in enumerate(CASES, 1):
            if index > 1:
                time.sleep(max(0, args.interval))
            mode, text, _, _ = case
            ctx = {'profile': dict(DEFAULTS), 'reference': DEFAULT_REFERENCE, 'analysis': None, 'history': [], 'mode': mode}
            system, user = build_prompts(text, ctx)
            result = {'id': index, 'mode': mode, 'request': text, 'expectation': case[3], 'status': 'ERROR'}
            started = time.monotonic()
            try:
                response = client.post(os.environ['LLM_BASE_URL'].rstrip('/') + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + os.environ['LLM_API_KEY']},
                    json={'model': os.environ['LLM_MODEL'], 'temperature': 0, 'max_tokens': 800,
                          'response_format': {'type': 'json_object'},
                          'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
                result['http_status'] = response.status_code
                body = response.json()
                if response.status_code != 200:
                    result['error'] = body.get('error', {}).get('message', 'Provider error')
                    result['retry_after'] = response.headers.get('retry-after')
                else:
                    choice = body['choices'][0]
                    result['usage'] = body.get('usage')
                    result['raw_answer'] = choice['message']['content']
                    if choice['finish_reason'] != 'stop':
                        raise ValueError('Incomplete AI answer: ' + choice['finish_reason'])
                    obj = json.loads(result['raw_answer'])
                    answer = check_answer(obj, text, ctx['profile'], mode)
                    result['answer'] = answer
                    result['status'] = 'PASS' if assess(answer, case) else 'FAIL'
            except Exception as exc:
                # Avoid dumping requests, headers, keys, or arbitrary tracebacks.
                result['error'] = type(exc).__name__ + ': ' + str(exc)
            result['seconds'] = round(time.monotonic() - started, 2)
            results.append(result)
            output.write_text(json.dumps({**metadata, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print('Saved report: ' + str(output), flush=True)
    write_markdown(output.with_suffix('.md'), metadata, results)


if __name__ == '__main__':
    main()
