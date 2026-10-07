import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
ARGS = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"]


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(args=ARGS)
        yield b
        b.close()


@pytest.fixture(scope="session")
def dist_url():
    subprocess.run([sys.executable, str(ROOT / "scripts/build_web.py")], check=True)
    return (ROOT / "dist/voice-enhancer.html").as_uri()


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


def _start(cmd, port, cwd, env=None):
    import os
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    for _ in range(150):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close(); return proc
        except OSError:
            time.sleep(0.2)
    proc.kill(); raise RuntimeError("process did not start: " + proc.stdout.read().decode())


@pytest.fixture(scope="session")
def server_url(tmp_path_factory):
    port = _free_port()
    env = {"LLM_BASE_URL": "", "LLM_MODEL": "", "LLM_API_KEY": "", "ANTHROPIC_API_KEY": ""}
    proc = _start([sys.executable, "-m", "voice_engine.cli", "--env-file", "/dev/null", "serve", "--port", str(port), "--backend", "cpu"], port, ROOT / "server", env)
    yield f"http://127.0.0.1:{port}"
    proc.terminate(); proc.wait(10)


@pytest.fixture(scope="session")
def gateway_server_url(tmp_path_factory):
    """voice-engine wired to a fake OpenAI-compatible gateway through a .env file, exactly as in production."""
    gport, port = _free_port(), _free_port()
    gw = _start([sys.executable, "-m", "uvicorn", "fake_gateway:app", "--port", str(gport)], gport, Path(__file__).parent)
    env_file = tmp_path_factory.mktemp("env") / ".env"
    env_file.write_text(f"LLM_BASE_URL=http://127.0.0.1:{gport}/v1\nLLM_MODEL=test-model\nLLM_API_KEY=test-key\nLLM_THINKING_PARAM=chat_template_kwargs\n"
                        f"VOICE_DB_PATH={env_file.parent / 'voice.sqlite'}\n")
    srv = _start([sys.executable, "-m", "voice_engine.cli", "--env-file", str(env_file), "serve", "--port", str(port), "--backend", "cpu"], port, ROOT / "server",
                 {"LLM_BASE_URL": "", "LLM_MODEL": "", "LLM_API_KEY": ""})
    yield f"http://127.0.0.1:{port}", f"http://127.0.0.1:{gport}"
    srv.terminate(); gw.terminate(); srv.wait(10); gw.wait(10)


@pytest.fixture()
def page_factory(browser):
    ctxs = []

    def make(url, init_script=None, **ctx_kw):
        ctx = browser.new_context(permissions=["microphone"], viewport={"width": 1360, "height": 960}, **ctx_kw)
        if init_script:
            ctx.add_init_script(init_script)
        pg = ctx.new_page()
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" and "fonts.g" not in m.text and "403" not in m.text and "ERR_" not in m.text else None)
        pg.goto(url)
        pg.wait_for_function("window.__ve && window.__ve.ready")
        ctxs.append(ctx)
        return pg

    yield make
    for c in ctxs:
        c.close()
