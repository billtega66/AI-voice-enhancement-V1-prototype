# Testing

| Suite | Command | Count | What it proves |
| --- | --- | --- | --- |
| Browser self-test | `node tests/js/run_selftest.js` (also in the app: Mixer → Self-test) | 21 | Each browser DSP stage, analysis, interpreter and validation against known signals |
| Server tests | `cd server && pytest` | 63 | Server stages, API, WebSocket, CLI, backends, Claude interpreter (mocked client) |
| Parity | part of the server tests | 19 | Browser and server engines give the same audio (≤ 1e-5), speech decisions, schema, validation, interpreter output, reference matching and analysis |
| End-to-end | `pytest tests/e2e` | 4 | Real Chromium with a fake microphone, standalone and served |

## Mapping to the functional requirements (section 14)

| # | Requirement | Covered by |
| --- | --- | --- |
| 1 | Microphone capture | e2e `test_standalone_full_workflow` (record), `test_served_uses_server_backend` (live over WebSocket) |
| 2 | Recording | e2e record step; duration checked |
| 3 | Playback | e2e play step |
| 4 | Original/Enhanced comparison | e2e A/B switch while playing; self-test `align` |
| 5 | AI changes affect processing | e2e: profile changes and the rendered preview audio changes; `test_validation_matches`, `test_parse_ai_response_variants` |
| 6 | Mixer modifies parameters | e2e Mixer slider; `test_every_profile_parameter_changes_the_audio` |
| 7 | Mixer and AI share one profile | e2e "keep everything else" step; self-test `store`; `test_interpret_offline_and_keeps_profile` |
| 8 | Noise suppression processes audio | `test_noise_suppression_reduces_noise_keeps_speech` |
| 9 | VAD detects speech | `test_vad_detects_speech_but_not_keyboard_or_paper`; self-test `vad`, `vadmute` |
| 10 | EQ modifies audio | `test_eq_bands` |
| 11 | Compression modifies dynamic range | `test_compressor_narrows_dynamic_range` |
| 12 | De-essing affects the sibilant band | `test_deesser_reduces_only_sibilant_band` |
| 13 | Limiting and output gain | `test_limiter_ceiling_and_output_gain` |
| 14 | Preview regeneration | e2e render counter and audio comparison |
| 15 | Use voice saves and activates | e2e Use voice, live uses the active profile, survives reload |
| 16 | Bypass restores the original | `test_bypass_is_bit_exact`; e2e bypass in live mode (browser and server) |
| 17 | Errors for missing permissions or capabilities | e2e `test_standalone_reports_blocked_microphone`; API bad-input tests |

## Not covered here

- ROCm execution: no AMD GPU in CI. `test_torch_ops_contract_matches_numpy` checks the GPU code path's calls with a stand-in tensor library; `scripts/verify_rocm.py` must be run on the AMD machine.
- Quality of real Claude responses: the interpreter is tested with mocked model replies. Judge real responses with the user study.
- Perceptual quality: tests check signal properties, not how natural the result sounds. That belongs in the 5–8 person user study.
