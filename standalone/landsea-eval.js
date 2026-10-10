// 世界海陆地图评测 —— JavaScript 评分程序（浏览器 + Node），与 evaluator.py 逐字段一致。
(function (root) {
  const H = 90, W = 180, N = H * W, R_EARTH = 6371.0, D2R = Math.PI / 180;
  const WEIGHTS = { mcc: 40, tolerant_iou: 20, coastline_f1: 20, landmass: 20 };
  const LAT = new Float64Array(N), LON = new Float64Array(N), WT = new Float64Array(N);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const i = r * W + c, la = 89 - 2 * r, lo = -179 + 2 * c;
    LAT[i] = la; LON[i] = lo; WT[i] = Math.cos(la * D2R);
  }
  function psum(arr) { // numpy pairwise summation
    const n = arr.length;
    if (n < 8) { let s = 0; for (let i = 0; i < n; i++) s += arr[i]; return s; }
    if (n <= 128) {
      const r = [arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]];
      let i = 8;
      for (; i < n - (n % 8); i += 8) for (let k = 0; k < 8; k++) r[k] += arr[i + k];
      let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += arr[i];
      return res;
    }
    let n2 = Math.floor(n / 2); n2 -= n2 % 8;
    return psum(arr.subarray(0, n2)) + psum(arr.subarray(n2));
  }
  const wsum = (m) => { const t = new Float64Array(N); for (let i = 0; i < N; i++) t[i] = m[i] ? WT[i] : 0; return psum(t); };
  const and = (a, b) => { const o = new Uint8Array(N); for (let i = 0; i < N; i++) o[i] = a[i] & b[i]; return o; };
  const div = (a, b) => (b > 0 ? a / b : 0);
  function dilate(m) {
    const o = new Uint8Array(N);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      if (!m[r * W + c]) continue;
      for (let dr = -1; dr <= 1; dr++) { const rr = r + dr; if (rr < 0 || rr >= H) continue;
        for (let dc = -1; dc <= 1; dc++) o[rr * W + ((c + dc + W) % W)] = 1; }
    }
    return o;
  }
  function boundary(m) {
    const b = new Uint8Array(N);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const i = r * W + c; if (!m[i]) continue;
      if (!m[r * W + (c + 1) % W] || !m[r * W + (c - 1 + W) % W] || (r > 0 && !m[i - W]) || (r < H - 1 && !m[i + W])) b[i] = 1;
    }
    return b;
  }
  function components(m) {
    const seen = new Uint8Array(N), comps = [];
    for (let s = 0; s < N; s++) {
      if (!m[s] || seen[s]) continue;
      const stack = [s], cells = []; seen[s] = 1;
      while (stack.length) {
        const i = stack.pop(), r = Math.floor(i / W), c = i % W; cells.push(i);
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr; if (rr < 0 || rr >= H) continue;
          const j = rr * W + ((c + dc + W) % W);
          if (m[j] && !seen[j]) { seen[j] = 1; stack.push(j); }
        }
      }
      cells.sort((a, b) => a - b); comps.push(cells);
    }
    comps.sort((a, b) => b.length - a.length || a[0] - b[0]);
    return comps;
  }
  function hav(i, j) {
    const la1 = LAT[i] * D2R, la2 = LAT[j] * D2R, lo1 = LON[i] * D2R, lo2 = LON[j] * D2R;
    const s1 = Math.sin((la2 - la1) / 2), s2 = Math.sin((lo2 - lo1) / 2);
    let h = s1 * s1 + Math.cos(la1) * Math.cos(la2) * s2 * s2;
    h = Math.min(1, Math.max(0, h));
    return 2 * R_EARTH * Math.asin(Math.sqrt(h));
  }
  const rnd = (x, d) => { const f = Math.pow(10, d); return Math.round(x * f) / f; };

  function evaluate(pred, gt, modelName, inputFormat) {
    const P = new Uint8Array(N), G = new Uint8Array(N), nP = new Uint8Array(N), nG = new Uint8Array(N);
    for (let i = 0; i < N; i++) { P[i] = pred[i] ? 1 : 0; G[i] = gt[i] ? 1 : 0; nP[i] = 1 - P[i]; nG[i] = 1 - G[i]; }
    const tp = wsum(and(P, G)), fp = wsum(and(P, nG)), fn = wsum(and(nP, G)), tn = wsum(and(nP, nG));
    const tot = tp + fp + fn + tn;
    const den = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    const mcc = Math.max(0, div(tp * tn - fp * fn, den));
    const landIou = div(tp, tp + fp + fn), acc = div(tp + tn, tot);

    const dP = dilate(P), dG = dilate(G);
    const tolP = div(wsum(and(P, dG)), wsum(P)), tolR = div(wsum(and(G, dP)), wsum(G));
    const tolIou = div(tolP * tolR, tolP + tolR - tolP * tolR);

    const bP = boundary(P), bG = boundary(G);
    let nbP = 0, nbG = 0; for (let i = 0; i < N; i++) { nbP += bP[i]; nbG += bG[i]; }
    let cp, cr, coastF1;
    if (nbP === 0 || nbG === 0) { coastF1 = nbP === nbG ? 1 : 0; cp = cr = coastF1; }
    else {
      cp = div(wsum(and(bP, dilate(bG))), wsum(bP)); cr = div(wsum(and(bG, dilate(bP))), wsum(bG));
      coastF1 = div(2 * cp * cr, cp + cr);
    }

    const comps = components(G); let num = 0, dnm = 0; const top = [];
    comps.forEach((idx, k) => {
      const a = new Float64Array(idx.length), b = new Float64Array(idx.length);
      idx.forEach((j, t) => { a[t] = WT[j] * P[j]; b[t] = WT[j]; });
      const rec = div(psum(a), psum(b)), s = Math.sqrt(idx.length);
      num += s * rec; dnm += s;
      if (k < 12) { const r = Math.floor(idx[0] / W), c = idx[0] % W;
        top.push({ cells: idx.length, first_cell_lat_lon: [89 - 2 * r, -179 + 2 * c], recall: rnd(rec, 4) }); }
    });
    const lmRecall = div(num, dnm), landmass = lmRecall * tolP;

    const pi = [], gi = []; for (let i = 0; i < N; i++) { if (bP[i]) pi.push(i); if (bG[i]) gi.push(i); }
    let chamfer = Math.PI * R_EARTH;
    if (pi.length && gi.length) {
      const mP = new Float64Array(pi.length).fill(Infinity), mG = new Float64Array(gi.length).fill(Infinity);
      for (let a = 0; a < pi.length; a++) for (let b = 0; b < gi.length; b++) {
        const d = hav(pi[a], gi[b]); if (d < mP[a]) mP[a] = d; if (d < mG[b]) mG[b] = d;
      }
      chamfer = 0.5 * (psum(mP) / pi.length + psum(mG) / gi.length);
    }
    const pts = { mcc_pts: WEIGHTS.mcc * mcc, tolerant_iou_pts: WEIGHTS.tolerant_iou * tolIou,
      coastline_f1_pts: WEIGHTS.coastline_f1 * coastF1, landmass_pts: WEIGHTS.landmass * landmass };
    const total = pts.mcc_pts + pts.tolerant_iou_pts + pts.coastline_f1_pts + pts.landmass_pts;
    let cellsP = 0, cellsG = 0; for (let i = 0; i < N; i++) { cellsP += P[i]; cellsG += G[i]; }
    const br = {}; for (const k in pts) br[k] = rnd(pts[k], 2);
    return {
      model_name: modelName, input_format: inputFormat || "180×90 PNG",
      composite_score_100: rnd(total, 2), official_score_breakdown: br,
      metrics: { mcc: rnd(mcc, 4), tolerant_iou: rnd(tolIou, 4), tolerant_precision: rnd(tolP, 4), tolerant_recall: rnd(tolR, 4),
        coastline_f1: rnd(coastF1, 4), coastline_precision: rnd(cp, 4), coastline_recall: rnd(cr, 4),
        landmass_recall: rnd(lmRecall, 4), landmass_score: rnd(landmass, 4) },
      diagnostics: { weighted_accuracy: rnd(acc, 4), land_iou: rnd(landIou, 4),
        land_share_pred: rnd(div(wsum(P), tot), 4), land_share_gt: rnd(div(wsum(G), tot), 4),
        land_cells_pred: cellsP, land_cells_gt: cellsG, coastline_chamfer_km: rnd(chamfer, 1), gt_landmass_count: comps.length },
      largest_gt_landmasses: top,
      // 逐格对错图: 0=水域正确 1=陆地正确 2=多画的陆地 3=漏画的陆地
      error_map: Array.from(P, (p, i) => (p && G[i] ? 1 : p ? 2 : G[i] ? 3 : 0)).join(""),
    };
  }

  // ---- 输入解析（与 app.py 相同的规则与报错文案） ----
  function parseCSV(text) {
    const lines = text.replace(/^\uFEFF/, "").replace(/\r/g, "").split("\n");
    while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
    if (lines.length !== H) throw new Error(`CSV 必须恰好 90 行，当前 ${lines.length} 行。`);
    const m = new Uint8Array(N);
    for (let r = 0; r < H; r++) {
      const f = lines[r].split(",").map((s) => s.trim());
      if (f.length !== W) throw new Error(`CSV 第 ${r + 1} 行应有 180 个值，当前 ${f.length} 个。`);
      for (let c = 0; c < W; c++) {
        if (f[c] !== "0" && f[c] !== "1") throw new Error(`CSV 第 ${r + 1} 行第 ${c + 1} 列的值为 “${f[c]}”，只允许 0 或 1。`);
        m[r * W + c] = f[c] === "1" ? 1 : 0;
      }
    }
    return m;
  }
  // rgba: Uint8(Clamped)Array, width*height*4
  function parseRGBA(rgba, width, height) {
    if (!((width === 180 && height === 90) || (width === 1440 && height === 720)))
      throw new Error(`图片尺寸必须为 180×90 或 1440×720 像素，当前为 ${width}×${height}。`);
    const v = new Uint8Array(width * height); let bad = 0, ex = null;
    for (let i = 0; i < width * height; i++) {
      const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2], a = rgba[4 * i + 3];
      if (a === 255 && r === g && g === b && (r === 0 || r === 255)) v[i] = r === 255 ? 1 : 0;
      else { bad++; if (!ex) ex = [r, g, b, a]; }
    }
    if (bad) throw new Error(`图片只能包含纯黑 (0,0,0) 和纯白 (255,255,255) 不透明像素，发现 ${bad} 个其他像素，例如 RGBA(${ex.join(",")})。`);
    if (width === 180) return { matrix: v, format: "180×90 PNG" };
    const m = new Uint8Array(N); let nonUni = 0;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const x0 = v[(r * 8) * 1440 + c * 8];
      for (let dy = 0; dy < 8; dy++) for (let dx = 0; dx < 8; dx++) if (v[(r * 8 + dy) * 1440 + c * 8 + dx] !== x0) { nonUni++; dy = dx = 8; }
      m[r * W + c] = x0;
    }
    if (nonUni) throw new Error(`1440×720 图片不是 8 倍最近邻放大：有 ${nonUni} 个 8×8 块内部颜色不一致。`);
    return { matrix: m, format: "1440×720 PNG（最近邻 ×8 已校验）" };
  }
  async function evaluateFile(file, gt) {
    const name = file.name || "upload", low = name.toLowerCase();
    if (low.endsWith(".csv")) return evaluate(parseCSV(await file.text()), gt, name, "90×180 CSV");
    if (!low.endsWith(".png")) throw new Error("只支持提交 PNG 图片或 CSV 矩阵文件。");
    let bmp;
    try { bmp = await createImageBitmap(file, { colorSpaceConversion: "none", premultiplyAlpha: "none" }); }
    catch (e) { throw new Error("无法读取 PNG 图片: " + e.message); }
    const cv = document.createElement("canvas"); cv.width = bmp.width; cv.height = bmp.height;
    const ctx = cv.getContext("2d", { willReadFrequently: true }); ctx.drawImage(bmp, 0, 0);
    const { matrix, format } = parseRGBA(ctx.getImageData(0, 0, bmp.width, bmp.height).data, bmp.width, bmp.height);
    return evaluate(matrix, gt, name, format);
  }
  const api = { evaluate, evaluateFile, parseCSV, parseRGBA, WEIGHTS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.LandSeaEval = api;
})(typeof window !== "undefined" ? window : globalThis);
