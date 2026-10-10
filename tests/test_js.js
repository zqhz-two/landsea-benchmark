// node tests/test_js.js  —— JS 评分器 vs Python 评分器，逐字段对比
const fs = require("fs"), path = require("path");
const E = require("../standalone/landsea-eval.js");
const dir = path.join(__dirname, "fixtures");
const gt = E.parseCSV(fs.readFileSync(path.join(__dirname, "../data/ground_truth_landsea_2deg.csv"), "utf8"));
let bad = 0, n = 0;
for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".csv"))) {
  const exp = JSON.parse(fs.readFileSync(path.join(dir, f.replace(".csv", ".json")), "utf8"));
  const got = E.evaluate(E.parseCSV(fs.readFileSync(path.join(dir, f), "utf8")), gt, f, "90×180 CSV");
  const a = JSON.stringify(got), b = JSON.stringify(exp); n++;
  if (a !== b) { bad++; console.log("MISMATCH", f, "\n js:", a, "\n py:", b); } else console.log("ok", f, got.composite_score_100);
}
console.log(bad ? `${bad}/${n} 不一致` : `全部 ${n} 个样例逐字段一致`);
process.exit(bad ? 1 : 0);
