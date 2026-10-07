.PHONY: install schema web test test-py test-js test-e2e serve bench

install:            ## Python deps (CPU) + Playwright browser for e2e
	pip install -e "server[dev,llm]"
	python -m playwright install chromium

schema:             ## regenerate web/src/schema.generated.js from shared/*.json
	python scripts/gen_schema.py

web:                ## single-file build -> dist/voice-enhancer.html
	python scripts/build_web.py

test: test-js test-py test-e2e

test-js:            ## browser engine self-test in Node
	python scripts/gen_schema.py --check
	node tests/js/run_selftest.js

test-py:            ## server unit, API and browser/server parity tests
	cd server && python -m pytest -q

test-e2e:           ## Chromium end-to-end, standalone and served
	python -m pytest -q tests/e2e

serve:              ## http://127.0.0.1:8000
	cd server && python -m voice_engine.cli serve --backend auto

bench:
	cd server && python -m voice_engine.cli benchmark --csv ../benchmark.csv
