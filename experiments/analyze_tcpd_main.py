# -*- coding: utf-8 -*-
"""
analyze_tcpd_main.py — 从 results/raw/tcpd_main.json 生成主表与机理分析。

产出：
- results/processed/tcpd_main_table.csv   每算法平均 F1/Covering/P/R/k/时间
- results/processed/tcpd_per_dataset.csv  每数据集×算法明细（SC-PELT 为 20 种子均值）
- 控制台：机理要点（幻影影响、过分割统计、zero 基线对照）
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw" / "tcpd_main.json"
OUTD = ROOT / "results" / "processed"
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))  # for `import tcpd_data` below

rows = json.loads(RAW.read_text())

# ------------------------------------------------- 每算法聚合（SC-PELT 按种子平均后聚合）
by_alg = defaultdict(lambda: defaultdict(list))
per_ds = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (r["dataset"], r["algorithm"])
    for f in ["f1", "precision", "recall", "covering", "k", "runtime_s"]:
        per_ds[key][f].append(r[f])

alg_names = sorted({r["algorithm"] for r in rows})
print("=" * 100)
print(f"{'algorithm':16s} {'F1':>7s} {'Cover':>7s} {'Prec':>7s} {'Rec':>7s} {'k':>7s} {'time_s':>9s}")
summary = {}
for alg in alg_names:
    ds_stats = defaultdict(list)
    for (ds, a), st in per_ds.items():
        if a == alg:
            for f in ["f1", "precision", "recall", "covering", "k", "runtime_s"]:
                ds_stats[f].append(np.mean(st[f]))
    m = {f: (float(np.mean(v)), float(np.std(v))) for f, v in ds_stats.items()}
    summary[alg] = m
    print(f"{alg:16s} {m['f1'][0]:7.4f} {m['covering'][0]:7.4f} {m['precision'][0]:7.4f} "
          f"{m['recall'][0]:7.4f} {m['k'][0]:7.2f} {m['runtime_s'][0]:9.4f}")

# ------------------------------------------------- CSV 输出
def wcsv(path, header, rows_):
    p = ROOT / path
    with open(p, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows_:
            f.write(",".join(str(x) for x in r) + "\n")
    print("saved", p)

wcsv(OUTD / "tcpd_main_table.csv",
     ["algorithm", "f1_mean", "f1_std", "covering_mean", "covering_std",
      "precision_mean", "recall_mean", "k_mean", "runtime_s_mean"],
     [[a, summary[a]["f1"][0], summary[a]["f1"][1], summary[a]["covering"][0],
       summary[a]["covering"][1], summary[a]["precision"][0], summary[a]["recall"][0],
       summary[a]["k"][0], summary[a]["runtime_s"][0]] for a in alg_names])

per_rows = []
for (ds, a), st in sorted(per_ds.items()):
    per_rows.append([ds, a, np.mean(st["f1"]), np.mean(st["covering"]),
                     np.mean(st["precision"]), np.mean(st["recall"]),
                     np.mean(st["k"]), np.mean(st["runtime_s"])])
wcsv(OUTD / "tcpd_per_dataset.csv",
     ["dataset", "algorithm", "f1_mean", "covering_mean", "precision_mean",
      "recall_mean", "k_mean", "runtime_s_mean"], per_rows)

# ------------------------------------------------- 机理要点（控制台 + notes）
print("=" * 100)
# M2: 过分割统计 —— k 与标注员平均 k 的对比
ann_counts = {}
import tcpd_data as D
for ds in sorted({r["dataset"] for r in rows}):
    ann = D.load_annotations(ds)
    ann_counts[ds] = np.mean([len(v) for v in ann.values()])
over = defaultdict(int)
for (ds, a), st in per_ds.items():
    k_mean = np.mean(st["k"])
    if k_mean > ann_counts[ds] + 1:
        over[a] += 1
print("过分割数据集数（k_mean > 标注均值+1）：", dict(over))
print("标注员平均变点数（全体中位）:", float(np.median(list(ann_counts.values()))))
