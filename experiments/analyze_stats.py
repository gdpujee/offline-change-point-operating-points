# -*- coding: utf-8 -*-
"""
analyze_stats.py — 统计检验（skill §38）。

1) TCPD：sc_pelt（20 种子逐数据集均值）vs 各 baseline 的逐数据集配对比较
   - Wilcoxon 符号秩检验（scipy，双侧）
   - win/loss/tie 计数（F1 差 > 1e-9）
2) 合成：各算法 P/R/kerr 按 noise×σ 汇总（供正文表）
输出：results/processed/stats_tcpd.csv, stats_synth.csv
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
OUTD = ROOT / "results" / "processed"
OUTD.mkdir(parents=True, exist_ok=True)


def main():
    rows = json.loads((ROOT / "results" / "raw" / "tcpd_main.json").read_text())
    per_ds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_ds[(r["dataset"], r["algorithm"])]["f1"].append(r["f1"])
        per_ds[(r["dataset"], r["algorithm"])]["covering"].append(r["covering"])

    datasets = sorted({ds for ds, _ in per_ds})
    sc = {ds: np.mean(per_ds[(ds, "sc_pelt")]["f1"]) for ds in datasets}
    out = ["algorithm,mean_f1_diff,win,loss,tie,wilcoxon_p"]
    print("=== TCPD: sc_pelt vs baselines（F1，配对）===")
    for alg in ["amoc", "binseg", "pelt", "segneigh", "zero", "pelt_bic"]:
        diffs = []
        for ds in datasets:
            base = per_ds[(ds, alg)]["f1"][0] if alg != "sc_pelt" else sc[ds]
            diffs.append(sc[ds] - base)
        diffs = np.array(diffs)
        w, p = st.wilcoxon(diffs) if np.any(np.abs(diffs) > 1e-12) else (np.nan, 1.0)
        win = int(np.sum(diffs > 1e-9))
        loss = int(np.sum(diffs < -1e-9))
        tie = len(diffs) - win - loss
        print(f"{alg:16s} ΔF1={diffs.mean():+.4f} win/loss/tie={win}/{loss}/{tie} Wilcoxon p={p:.4g}")
        out.append(f"{alg},{diffs.mean():.6f},{win},{loss},{tie},{p:.6g}")
    (OUTD / "stats_tcpd.csv").write_text("algorithm," + "\n".join(l for l in "\n".join(out).split(",")[1:]) if False else "\n".join(out) + "\n")

    # 合成汇总
    srows = json.loads((ROOT / "results" / "raw" / "synth_grid.json").read_text())
    agg = defaultdict(list)
    for r in srows:
        agg[(r["algorithm"], r["noise"], r["sigma"])].append(
            (r["precision"], r["recall"], r["k_pred"] - r["k_true"]))
    lines = ["algorithm,noise,sigma,precision,recall,kerr"]
    for (a, nz, s), v in sorted(agg.items()):
        lines.append(f"{a},{nz},{s},"
                     f"{np.mean([x[0] for x in v]):.4f},"
                     f"{np.mean([x[1] for x in v]):.4f},"
                     f"{np.mean([x[2] for x in v]):.4f}")
    (OUTD / "stats_synth.csv").write_text("\n".join(lines) + "\n")
    print("saved", OUTD / "stats_tcpd.csv", OUTD / "stats_synth.csv")


if __name__ == "__main__":
    main()
