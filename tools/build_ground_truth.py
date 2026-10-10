"""从 Natural Earth 10m 标准海陆数据生成 90×180 海陆真值（1=陆地，0=水域）。

运行: pip install numpy shapely pyshp pillow
      python tools/build_ground_truth.py      # 自动下载 Natural Earth 5.1.1 数据

规则（与赛题定义一致）:
  * 采样点: 纬度 89,87,…,-89（90 行，北→南）；经度 -179,-177,…,179（180 列，西→东）；格点中心点判定。
  * 陆地 = ne_10m_land 多边形（大陆、岛屿，含格陵兰与南极接地冰盖）。
  * 内陆湖泊算陆地: ne_10m_land 本身不挖湖洞；唯一挖洞的是里海，这里补回为陆地。
  * 南极浮动冰架（ne_10m_antarctic_ice_shelves_polys）与海冰算水域: ne_10m_land 已不含冰架。
"""
import hashlib, io, urllib.request, zipfile
from pathlib import Path
import numpy as np, shapefile
from PIL import Image
from shapely import contains_xy
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tools" / "_ne_cache"
URL = "https://naciscdn.org/naturalearth/10m/physical/{}.zip"


def load(name):
    CACHE.mkdir(exist_ok=True)
    if not (CACHE / f"{name}.shp").exists():
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL.format(name)).read())).extractall(CACHE)
    r = shapefile.Reader(str(CACHE / name))
    return unary_union([shape(s.__geo_interface__) for s in r.shapes()])


def main():
    land = load("ne_10m_land")
    shelves = load("ne_10m_antarctic_ice_shelves_polys")
    # 填补内陆水体洞（实测只有里海一个）
    filled = unary_union([Polygon(p.exterior) for p in land.geoms])
    lat = 89 - 2 * np.arange(90)
    lon = -179 + 2 * np.arange(180)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    gt = contains_xy(filled, LO, LA)
    gt &= ~contains_xy(shelves, LO, LA)  # 保险: 冰架一律为水
    gt = gt.astype(np.uint8)
    np.save(ROOT / "data/ground_truth_landsea_2deg.npy", gt)
    np.savetxt(ROOT / "data/ground_truth_landsea_2deg.csv", gt, fmt="%d", delimiter=",")
    img = Image.fromarray(gt * 255, "L")
    img.save(ROOT / "data/ground_truth_180x90.png")
    img.resize((1440, 720), Image.NEAREST).save(ROOT / "data/ground_truth_1440x720.png")
    print("land cells:", int(gt.sum()), "/ 16200")
    print("sha256:", hashlib.sha256(gt.tobytes()).hexdigest())


if __name__ == "__main__":
    main()
