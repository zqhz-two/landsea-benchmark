"""生成参考榜单（基线）。新增模型成绩: 在 ENTRIES 里加一行 (名称, 类型, CSV/PNG 路径) 后重跑。"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator import evaluate, parse_csv, parse_rgba, LATG
gt = np.load(ROOT / "data/ground_truth_landsea_2deg.npy")
rng = np.random.default_rng(2026)
BASELINES = [
    ("基线 · 全部画成陆地", np.ones_like(gt)),
    ("基线 · 随机 30% 陆地", (rng.random(gt.shape) < 0.3).astype(np.uint8)),
    ("基线 · 只画南极（≤65°S）", (LATG <= -65).astype(np.uint8)),
    ("基线 · 全部画成水域", np.zeros_like(gt)),
]
ENTRIES = []  # 例: ("某模型", "模型", "submissions/xxx.csv")
rows = []
for name, m in BASELINES:
    rows.append((name, "基线", evaluate(m, gt, name)))
for name, kind, p in ENTRIES:
    p = ROOT / p
    m = parse_csv(p.read_text("utf-8")) if p.suffix == ".csv" else parse_rgba(np.array(Image.open(p).convert("RGBA")))[0]
    rows.append((name, kind, evaluate(m, gt, name)))
rows.sort(key=lambda t: -t[2]["composite_score_100"])
lb = []
for i, (name, kind, r) in enumerate(rows, 1):
    lb.append({"rank": i, "model_name": name, "kind": kind, "composite_score": r["composite_score_100"],
               "mcc": r["metrics"]["mcc"], "tolerant_iou": r["metrics"]["tolerant_iou"],
               "coastline_f1": r["metrics"]["coastline_f1"], "landmass_score": r["metrics"]["landmass_score"],
               "coastline_chamfer_km": r["diagnostics"]["coastline_chamfer_km"],
               "weighted_accuracy": r["diagnostics"]["weighted_accuracy"]})
out = {"benchmark_version": "LANDSEA-NE10M-5.1.1-2DEG-v1",
       "ground_truth_sha256": hashlib.sha256(gt.astype(np.uint8).tobytes()).hexdigest(),
       "updated_at": "2026-10-11", "scoring_weights": {"mcc": 40, "tolerant_iou": 20, "coastline_f1": 20, "landmass": 20},
       "leaderboard": lb}
(ROOT / "data/leaderboard.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
for e in lb: print(e["rank"], e["model_name"], e["composite_score"])
