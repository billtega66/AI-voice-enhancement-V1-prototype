#!/usr/bin/env python3
"""Bundle web/ into one self-contained HTML file (dist/voice-enhancer.html) that runs
from file:// or any static host (it is also what gets published as a Claude artifact)."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT / "scripts/gen_schema.py")], check=True)
html = (ROOT / "web/index.html").read_text(encoding='utf-8')


def inline(m):
    js = (ROOT / "web" / m.group(1)).read_text(encoding='utf-8')
    return "<script>\n" + js.replace("</script", "<\\/script") + "\n</script>"


out = re.sub(r'<script src="([^"]+)"></script>', inline, html)
dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
(dist / "voice-enhancer.html").write_text(out, encoding='utf-8')
print(f"wrote dist/voice-enhancer.html ({len(out) // 1024} KB)")
