# Modular voice challenges

| Request | Result | AI reply | Changes |
|---|---|---|---|
| Make my voice sound like a broken alien transmission: metallic, gritty, with short repeating echoes. Keep my words understandable. | PASS | Sure, adding a metallic texture, some grit, and short echoes. | {"metallicMix": 0.3, "metallicHz": 80.0, "distortionDrive": 0.3, "echoMix": 0.3, "echoMs": 120, "echoFeedback": 0.3, "limiterEnabled": true} |
| Make it sound like I am speaking inside a huge stone cavern. If you cannot produce real room reverberation, tell me and offer an honest approximation. | ERROR | The AI provider has reached a request or token limit. Wait before trying again; daily limits may require waiting for a reset. | {} |
| Make me sound like a tiny cartoon creature, but keep my pitch exactly unchanged. Ask me before approximating it with something else. | PASS | Do you want a metallic texture, distortion, or echo? | {} |
| Make me speak slowly and sleepily, with longer pauses between words. Preserve every word. | PASS | I can't slow down speech or add pauses. | {} |
| Remove the metallic texture and echo. Keep my existing warmth and pitch exactly as they are. | FAIL | This engine cannot produce the full requested effect. Missing capability: metallic. | {} |
| Keep my pitch exactly unchanged, but raise my pitch by one octave at the same time. | PASS | What do you mean by keeping pitch unchanged while raising it? | {} |
| Give me the exact voice, accent, and speaking rhythm of a completely different person, while keeping the words from my recording. | PASS | I can't change your voice to match another person. | {} |
| Make my voice a little brighter and soften the sharp S sounds. Do not change pitch, add echo, or add a metallic texture. | ERROR | The AI gateway returned HTTP 400. | {} |
