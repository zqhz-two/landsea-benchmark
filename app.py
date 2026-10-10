"""世界海陆地图评测基准 —— 服务器版。

运行:  pip install -r requirements.txt
       uvicorn app:app --host 0.0.0.0 --port 8000
接口:  GET /   GET /api/prompt   GET /api/leaderboard   POST /api/evaluate（字段名 file，PNG 或 CSV）
"""
import hashlib, io, json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from evaluator import InputError, evaluate, parse_csv, parse_rgba

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GT = np.load(DATA / "ground_truth_landsea_2deg.npy").astype(np.uint8)
GT_SHA256 = hashlib.sha256(GT.tobytes()).hexdigest()

app = FastAPI(title="World Land-Sea Map Benchmark", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse(ROOT / "static" / "index.html", media_type="text/html")


@app.get("/api/prompt", response_class=PlainTextResponse)
def serve_prompt():
    return (DATA / "prompt.txt").read_text(encoding="utf-8")


@app.get("/api/leaderboard")
def serve_leaderboard():
    return JSONResponse(json.loads((DATA / "leaderboard.json").read_text(encoding="utf-8")))


@app.post("/api/evaluate")
async def evaluate_upload(file: UploadFile = File(...)):
    name = file.filename or "upload"
    raw = await file.read()
    try:
        if name.lower().endswith(".csv"):
            m, fmt = parse_csv(raw.decode("utf-8")), "90×180 CSV"
        elif name.lower().endswith(".png"):
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGBA")
            except Exception as e:
                raise InputError(f"无法读取 PNG 图片: {e}")
            m, fmt = parse_rgba(np.array(img))
        else:
            raise InputError("只支持提交 PNG 图片或 CSV 矩阵文件。")
    except (InputError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    out = evaluate(m, GT, model_name=name, input_format=fmt)
    out["ground_truth_sha256"] = GT_SHA256
    return out
