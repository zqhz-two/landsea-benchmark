# 世界海陆地图评测基准（Land-Sea Map Benchmark）

基于「柯本世界模型评测基准」备份改造：题目换成“仅凭已有地理知识重建世界海陆地图”，
真值换成 Natural Earth 标准海陆数据，评分规则改为专门针对 0/1 海陆图设计。

## 两种用法
1. **离线单文件版**：双击 `standalone/index.html`。题目、榜单、真值、评分程序都在这一个文件里，可直接放到任意静态托管。
2. **服务器版**：
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   接口：`GET /`、`GET /api/prompt`、`GET /api/leaderboard`、`POST /api/evaluate`（字段名 `file`，PNG 或 CSV）。

## 可提交的文件
| 格式 | 要求 |
| --- | --- |
| `.csv` | 90 行 × 180 列，无表头，只能是 0/1（1=陆地） |
| `.png` 180×90 | 只能有纯黑 (0,0,0) 与纯白 (255,255,255) 不透明像素；白=陆地 |
| `.png` 1440×720 | 同上，且必须是 8 倍最近邻放大（每个 8×8 块颜色一致），否则拒收 |

## 真值
- 来源：Natural Earth 10m Physical `ne_10m_land` v5.1.1（公有领域）。
- 在 16,200 个格心点（纬度 89…-89，经度 -179…179）做点在多边形判定。
- 按赛题定义：内陆湖泊算陆地（Natural Earth 陆地层只给里海挖了洞，已补回为陆地）；南极浮动冰架与海冰算水域（陆地层本身不含冰架，另用 `ne_10m_antarctic_ice_shelves_polys` 再扣一次保险）。
- 结果：5,391 个陆地格。SHA-256 `43e08bda7b99274a55be6efdc9f81b14f498b22e5a240acffb4863b98bedfd2d`。
- 文件：`data/ground_truth_landsea_2deg.npy/.csv`、`data/ground_truth_180x90.png`、`data/ground_truth_1440x720.png`。
- 重新生成：`pip install shapely pyshp` 后运行 `python tools/build_ground_truth.py`（已附带数据缓存，无需联网）。

## 评分（总分 100，所有面积量按 cos(纬度) 加权）
| 项目 | 分值 | 含义 |
| --- | --- | --- |
| 海陆相关 MCC | 40 | Matthews 相关系数，负值记 0。全画陆地、全画海洋、随机乱画都≈0 分，比 IoU 更难被投机 |
| 空间容错 IoU | 20 | 允许 ±1 格（3×3 邻域，经度首尾相接）错位的陆地 IoU |
| 海岸线 F1 | 20 | 海岸格（陆地格且 4 邻域有水）的 F1，允许 ±1 格偏差 |
| 陆块识别 | 20 | 真值中 42 个连通陆块（8 邻域）各自被画出的比例，按 √格数 加权平均，再乘容错精确率，防止“全画陆地”刷分。漏掉马达加斯加、日本、新西兰等岛屿会明显扣分 |

诊断项：加权准确率、陆地 IoU、陆地面积占比（作答/真值）、海岸线平均偏差 km（双向 Chamfer，球面距离）。
评分后页面显示逐格对错图：灰=陆地正确，黑=水域正确，红=多画的陆地，蓝=漏画的陆地。

参考：真值 100；经度整体偏 1 格（2°）≈ 89.8；去掉所有小于 30 格的岛屿 ≈ 90.8；随机翻转 5% 格子 ≈ 85.9；
基线：随机 30% 陆地 18.22，全画陆地 16.06，只画南极 8.55，全画水域 0。

## 本次对比榜
页面下方的榜单会把这次打开页面后上传的每个文件自动加一行、按总分排名，四条基线以灰色参考线显示。
文件本身不带模型信息，默认用文件名当名称，点名称即可改成模型名。关闭页面即清空，可用“下载 CSV”或“复制表格”（粘贴到 Excel）导出。

## 加入固定榜单
把模型交付的 CSV/PNG 放进 `submissions/`，在 `tools/build_leaderboard.py` 的 `ENTRIES` 里加一行，然后：
```bash
python tools/build_leaderboard.py && python tools/build_standalone.py
```

## 校验
```bash
python tests/make_fixtures.py   # 重新生成 13 个扰动样例的期望结果
python tests/test_py.py         # Python 评分器
node tests/test_js.js           # 浏览器 JS 评分器，与 Python 逐字段一致
```
