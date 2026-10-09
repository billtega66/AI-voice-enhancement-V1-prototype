# Modular voice challenges

| Request | Result | AI reply | Changes |
|---|---|---|---|
| Make it sound like I am speaking inside a huge stone cavern. If you cannot produce real room reverberation, tell me and offer an honest approximation. | FAIL | Sure, I'll add echo to simulate a cavern. | {"echoMix": 0.3, "echoMs": 500, "echoFeedback": 0.5, "limiterEnabled": true} |
| Remove the metallic texture and echo. Keep my existing warmth and pitch exactly as they are. | FAIL | This engine cannot produce the full requested effect. Missing capability: echoMix, metallicMix. | {} |
| Make my voice a little brighter and soften the sharp S sounds. Do not change pitch, add echo, or add a metallic texture. | PASS | Sure, I'll brighten the voice a bit and soften the sharp S sounds. | {"presenceDb": 1.0, "airDb": 1.0, "deEssMaxDb": 8.0} |
