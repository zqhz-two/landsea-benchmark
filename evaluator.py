"""世界海陆地图评测 —— 评分程序（Python 版，与 standalone/landsea-eval.js 逐字段一致）。

输入: 90×180 的 0/1 矩阵（1=陆地，0=水域；行=纬度 89→-89，列=经度 -179→179）。
总分 100:
  海陆相关性 MCC ×40   面积加权（cos 纬度）的 Matthews 相关系数，<0 记 0；全陆/全海/随机 ≈ 0 分。
  空间容错 IoU ×20      允许 ±1 格（3×3 邻域，经度首尾相接）错位的陆地 IoU。
  海岸线 F1 ×20         海陆分界格（4 邻域）的 F1，允许 ±1 格偏差，面积加权。
  陆块识别 ×20          真值中每个连通陆块（8 邻域）被画出的比例，按 √格数 加权平均，再乘容错精确率（防止“全画陆地”刷分）。
诊断: 加权准确率、陆地 IoU、陆地面积占比、海岸线平均偏差 km（双向 Chamfer）。
"""
import numpy as np

R_EARTH = 6371.0
H, W = 90, 180
LAT = (89 - 2 * np.arange(H)).astype(float)
LON = (-179 + 2 * np.arange(W)).astype(float)
LATG, LONG = np.meshgrid(LAT, LON, indexing="ij")
WT = np.cos(np.deg2rad(LATG))
WEIGHTS = {"mcc": 40.0, "tolerant_iou": 20.0, "coastline_f1": 20.0, "landmass": 20.0}


def dilate(m):
    """3×3 膨胀；经度首尾相接，两极不相接。"""
    out = m.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            s = np.roll(m, dc, axis=1)
            if dr == -1:
                s = np.vstack([s[1:], np.zeros((1, W), bool)])
            elif dr == 1:
                s = np.vstack([np.zeros((1, W), bool), s[:-1]])
            out |= s
    return out


def boundary(m):
    """陆地格中，4 邻域里有水域的格（海岸格）。"""
    m = m.astype(bool)
    nb = ~np.roll(m, 1, 1) | ~np.roll(m, -1, 1)
    nb[1:] |= ~m[:-1]
    nb[:-1] |= ~m[1:]
    return m & nb


def components(m):
    """8 邻域连通分量（经度首尾相接）。返回 [格索引数组, …]，按大小降序。"""
    seen = np.zeros(m.shape, bool)
    comps = []
    for r0, c0 in zip(*np.nonzero(m)):
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append(r * W + c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, (c + dc) % W
                    if 0 <= rr < H and m[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
        comps.append(np.array(sorted(cells)))
    comps.sort(key=lambda a: (-len(a), a[0]))
    return comps


def _hav(a_idx, b_idx):
    la1 = np.deg2rad(LATG.ravel()[a_idx])[:, None]
    lo1 = np.deg2rad(LONG.ravel()[a_idx])[:, None]
    la2 = np.deg2rad(LATG.ravel()[b_idx])[None, :]
    lo2 = np.deg2rad(LONG.ravel()[b_idx])[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _div(a, b):
    return float(a / b) if b > 0 else 0.0


def evaluate(pred, gt, model_name="upload", input_format="180×90 PNG"):
    P = np.asarray(pred).astype(bool)
    G = np.asarray(gt).astype(bool)
    w = WT
    tp = (w * (P & G)).sum(); fp = (w * (P & ~G)).sum()
    fn = (w * (~P & G)).sum(); tn = (w * (~P & ~G)).sum()
    tot = tp + fp + fn + tn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = max(0.0, _div(tp * tn - fp * fn, den))
    land_iou = _div(tp, tp + fp + fn)
    acc = _div(tp + tn, tot)

    dP, dG = dilate(P), dilate(G)
    tol_p = _div((w * (P & dG)).sum(), (w * P).sum())
    tol_r = _div((w * (G & dP)).sum(), (w * G).sum())
    tol_iou = _div(tol_p * tol_r, tol_p + tol_r - tol_p * tol_r)

    bP, bG = boundary(P), boundary(G)
    if bP.sum() == 0 or bG.sum() == 0:
        coast_f1 = 1.0 if bP.sum() == bG.sum() else 0.0
        cp = cr = coast_f1
    else:
        cp = _div((w * (bP & dilate(bG))).sum(), (w * bP).sum())
        cr = _div((w * (bG & dilate(bP))).sum(), (w * bG).sum())
        coast_f1 = _div(2 * cp * cr, cp + cr)

    flatP, flatw = P.ravel(), w.ravel()
    comps = components(G)
    num = dnm = 0.0
    top = []
    for k, idx in enumerate(comps):
        rec = _div((flatw[idx] * flatP[idx]).sum(), flatw[idx].sum())
        a = np.sqrt(len(idx))
        num += a * rec; dnm += a
        if k < 12:
            r, c = divmod(int(idx[0]), W)
            top.append({"cells": int(len(idx)), "first_cell_lat_lon": [float(LAT[r]), float(LON[c])], "recall": round(rec, 4)})
    lm_recall = _div(num, dnm)
    landmass = lm_recall * tol_p

    pi, gi = np.flatnonzero(bP.ravel()), np.flatnonzero(bG.ravel())
    if len(pi) and len(gi):
        d = _hav(pi, gi)
        chamfer = 0.5 * (d.min(1).mean() + d.min(0).mean())
    else:
        chamfer = np.pi * R_EARTH

    pts = {
        "mcc_pts": WEIGHTS["mcc"] * mcc,
        "tolerant_iou_pts": WEIGHTS["tolerant_iou"] * tol_iou,
        "coastline_f1_pts": WEIGHTS["coastline_f1"] * coast_f1,
        "landmass_pts": WEIGHTS["landmass"] * landmass,
    }
    return {
        "model_name": model_name,
        "input_format": input_format,
        "composite_score_100": round(sum(pts.values()), 2),
        "official_score_breakdown": {k: round(v, 2) for k, v in pts.items()},
        "metrics": {
            "mcc": round(mcc, 4),
            "tolerant_iou": round(tol_iou, 4),
            "tolerant_precision": round(tol_p, 4),
            "tolerant_recall": round(tol_r, 4),
            "coastline_f1": round(coast_f1, 4),
            "coastline_precision": round(cp, 4),
            "coastline_recall": round(cr, 4),
            "landmass_recall": round(lm_recall, 4),
            "landmass_score": round(landmass, 4),
        },
        "diagnostics": {
            "weighted_accuracy": round(acc, 4),
            "land_iou": round(land_iou, 4),
            "land_share_pred": round(_div((w * P).sum(), tot), 4),
            "land_share_gt": round(_div((w * G).sum(), tot), 4),
            "land_cells_pred": int(P.sum()),
            "land_cells_gt": int(G.sum()),
            "coastline_chamfer_km": round(float(chamfer), 1),
            "gt_landmass_count": len(comps),
        },
        "largest_gt_landmasses": top,
        # 逐格对错图: 0=水域正确 1=陆地正确 2=多画的陆地 3=漏画的陆地
        "error_map": "".join(map(str, np.where(P & G, 1, np.where(P & ~G, 2, np.where(~P & G, 3, 0))).ravel())),
    }


# ---------------- 输入解析（与 JS 版相同的规则与报错文案） ----------------
class InputError(ValueError):
    pass


def parse_csv(text):
    lines = text.lstrip("\ufeff").replace("\r", "").split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != H:
        raise InputError(f"CSV 必须恰好 90 行，当前 {len(lines)} 行。")
    m = np.zeros((H, W), np.uint8)
    for r, line in enumerate(lines):
        f = [s.strip() for s in line.split(",")]
        if len(f) != W:
            raise InputError(f"CSV 第 {r + 1} 行应有 180 个值，当前 {len(f)} 个。")
        for c, v in enumerate(f):
            if v not in ("0", "1"):
                raise InputError(f"CSV 第 {r + 1} 行第 {c + 1} 列的值为 “{v}”，只允许 0 或 1。")
            m[r, c] = v == "1"
    return m


def parse_rgba(rgba):
    """rgba: (h, w, 4) uint8。返回 (90×180 矩阵, 格式说明)。"""
    h, w = rgba.shape[:2]
    if (w, h) not in ((180, 90), (1440, 720)):
        raise InputError(f"图片尺寸必须为 180×90 或 1440×720 像素，当前为 {w}×{h}。")
    r, g, b, a = (rgba[..., i].astype(int) for i in range(4))
    ok = (a == 255) & (r == g) & (g == b) & ((r == 0) | (r == 255))
    bad = int((~ok).sum())
    if bad:
        ex = rgba.reshape(-1, 4)[np.flatnonzero(~ok.ravel())[0]]
        raise InputError(f"图片只能包含纯黑 (0,0,0) 和纯白 (255,255,255) 不透明像素，发现 {bad} 个其他像素，例如 RGBA({','.join(map(str, ex))})。")
    v = (r == 255).astype(np.uint8)
    if w == 180:
        return v, "180×90 PNG"
    blocks = v.reshape(H, 8, W, 8).transpose(0, 2, 1, 3).reshape(H, W, 64)
    non_uni = int((blocks != blocks[..., :1]).any(-1).sum())
    if non_uni:
        raise InputError(f"1440×720 图片不是 8 倍最近邻放大：有 {non_uni} 个 8×8 块内部颜色不一致。")
    return blocks[..., 0].copy(), "1440×720 PNG（最近邻 ×8 已校验）"
