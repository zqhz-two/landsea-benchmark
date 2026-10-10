"""python tests/test_py.py —— 重新评分全部样例，确认与保存的结果一致。"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator import evaluate, parse_csv
gt = np.load(ROOT / "data/ground_truth_landsea_2deg.npy")
bad = 0
for f in sorted((ROOT / "tests/fixtures").glob("*.csv")):
    got = evaluate(parse_csv(f.read_text()), gt, f.name, "90×180 CSV")
    exp = json.loads(f.with_suffix(".json").read_text())
    ok = json.dumps(got, ensure_ascii=False) == json.dumps(exp, ensure_ascii=False)
    bad += not ok
    print("ok" if ok else "MISMATCH", f.name, got["composite_score_100"])
print("全部一致" if not bad else f"{bad} 个不一致")
