# Voice AI evaluation

Model: `openai/gpt-oss-20b`

Each request starts from the default profile; no offline fallback is used.
PASS checks the expected parameter direction or no-change behavior, not audio quality.

| # | Mode | Request | Result | AI reply / error | Changes |
|---|---|---|---|---|---|
| 1 | enhancement | Make my voice warmer. | PASS | Sure, adding warmth. | {"warmthDb": 2.5} |
| 2 | enhancement | Reduce the sharp S sounds. | PASS | Sure, I'll reduce the sharp S sounds. | {"deEssThresholdDb": -40, "deEssMaxDb": 8.0} |
| 3 | enhancement | Remove keyboard noise while keeping my speech clear. | PASS | Sure, I'll reduce keyboard noise. | {"nsAmount": 0.8} |
| 4 | enhancement | Keep my volume steady without changing my pitch. | PASS | Sure, I've tightened the compression a bit. | {"compRatio": 3.0} |
| 5 | enhancement | Make it less bright, but keep the warmth. | PASS | Sure, I'll reduce the high frequencies a bit. | {"airDb": -1.0} |
| 6 | enhancement | Make my voice slightly deeper. | FAIL | Sure, I'll add a bit of warmth to deepen your voice. | {"warmthDb": 1.0} |
| 7 | creative | I want my voice to sound like chipmunks. | PASS | Sure, your voice will sound like a chipmunk. | {"pitchSemitones": 3.0} |
| 8 | enhancement | I want my voice to sound like chipmunks. | PASS | Select Creative mode to try character or exaggerated voice effects. | {} |
| 9 | creative | Make me sound like a robot. | FAIL | Sure, shifting your pitch up a bit. | {"pitchSemitones": 3.0} |
| 10 | creative | Make my voice different. | PASS | What kind of change would you like? | {} |
