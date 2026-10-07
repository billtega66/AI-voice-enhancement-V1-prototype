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


@pytest.fixture(scope="session")
def server_url():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen([sys.executable, "-m", "voice_engine.cli", "serve", "--port", str(port), "--backend", "cpu"],
                            cwd=ROOT / "server", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close(); break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill(); raise RuntimeError("server did not start: " + proc.stdout.read().decode())
    yield url
    proc.terminate(); proc.wait(10)


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
