"""生成测试样例（各种扰动），用于 Python 与 JS 评分器逐字段对比。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluator import evaluate, LATG
ROOT = Path(__file__).resolve().parents[1]
gt = np.load(ROOT / "data/ground_truth_landsea_2deg.npy")
rng = np.random.default_rng(7)
cases = {
    "gt": gt, "all_water": np.zeros_like(gt), "all_land": np.ones_like(gt),
    "shift_lon1": np.roll(gt, 1, 1), "shift_lon5": np.roll(gt, 5, 1), "shift_lat1": np.roll(gt, 1, 0),
    "flip_ns": gt[::-1], "noise5": gt ^ (rng.random(gt.shape) < 0.05), "noise20": gt ^ (rng.random(gt.shape) < 0.2),
    "random30": rng.random(gt.shape) < 0.3, "antarctic_only": LATG <= -65,
    "no_islands": None, "single_cell": np.zeros_like(gt),
}
from evaluator import components
nm = np.zeros(gt.shape, bool)
for c in components(gt.astype(bool)):
    if len(c) >= 30: nm.ravel()[c] = True
cases["no_islands"] = nm
cases["single_cell"][45, 90] = 1
out = ROOT / "tests/fixtures"; out.mkdir(exist_ok=True)
for k, m in cases.items():
    m = np.asarray(m).astype(np.uint8)
    np.savetxt(out / f"{k}.csv", m, fmt="%d", delimiter=",")
    json.dump(evaluate(m, gt, model_name=f"{k}.csv", input_format="90×180 CSV"), open(out / f"{k}.json", "w"), ensure_ascii=False, indent=1)
print(len(cases), "fixtures")
