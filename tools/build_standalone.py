"""生成单文件离线版 standalone/index.html（双击即可打开，也可放到任意静态托管）。"""
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "static/index.html").read_text(encoding="utf-8")
tw = (ROOT / "static/vendor/tailwindcss.js").read_text(encoding="utf-8")
ev = (ROOT / "standalone/landsea-eval.js").read_text(encoding="utf-8")
prompt = (ROOT / "data/prompt.txt").read_text(encoding="utf-8")
lb = json.loads((ROOT / "data/leaderboard.json").read_text(encoding="utf-8"))
gt = "".join(map(str, np.load(ROOT / "data/ground_truth_landsea_2deg.npy").ravel()))
html = html.replace('<script src="/static/vendor/tailwindcss.js"></script>', "<script>" + tw.replace("</script>", "<\\/script>") + "</script>")
data = ("<script>" + ev + "</script>\n<script>\n"
        f"window.__PROMPT__ = {json.dumps(prompt, ensure_ascii=False)};\n"
        f"window.__LEADERBOARD__ = {json.dumps(lb, ensure_ascii=False)};\n"
        f"window.__GT__ = Uint8Array.from({json.dumps(gt)}, c => +c);\n</script>\n")
html = html.replace("  <script>\n    // 1. Load Prompt", data + "  <script>\n    // 1. Load Prompt", 1)
html = html.replace("fetch('/api/prompt')\n      .then(res => res.text())", "Promise.resolve(window.__PROMPT__)")
html = html.replace("fetch('/api/leaderboard')\n      .then(res => res.json())", "Promise.resolve(window.__LEADERBOARD__)")
old = re.search(r"      fetch\('/api/evaluate', \{.*?\n      \}\)\n      \.then\(res => \{.*?\n      \}\)\n", html, re.S)
assert old, "evaluate block not found"
html = html.replace(old.group(0), "      LandSeaEval.evaluateFile(file, window.__GT__)\n")
html = html.replace("      const formData = new FormData();\n      formData.append('file', file);\n", "")
left = re.findall(r".{40}/api/.{30}", html)
assert not left, left
(ROOT / "standalone/index.html").write_text(html, encoding="utf-8")
print("written", len(html))
