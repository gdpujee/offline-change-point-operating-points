# -*- coding: utf-8 -*-
"""landscape.py — 无 oracle 参数景观：F1/Covering 随 β 常数、Q 容量、规则变体的变化。

目的：找设计空间地形，为算法 v4 提供依据（所有数值均可追溯到 raw）。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import metrics_tcpdbench as M  # noqa: E402
import tcpd_data as D  # noqa: E402
from run_tcpd_main import binseg_textbook  # noqa: E402


def main():
    rows = []
    datasets = []
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except FileNotFoundError:
            continue
        if np.all(np.isfinite(y)):
            datasets.append((ds, y, n, D.load_annotations(ds)))

    logns = {ds: np.log(n) for ds, y, n, a in datasets}

    # (i) 常数 β PELT 扫描
    for c in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        for ds, y, n, ann in datasets:
            cps = L.pelt(y, beta=c * np.log(n))
            f1, p, r = M.f_measure(ann, cps.tolist(), return_PR=True)
            rows.append({"family": "pelt_const", "param": c, "dataset": ds,
                         "k": len(cps), "f1": f1, "covering": M.covering(ann, cps.tolist(), n)})
        print(f"pelt_const c={c} done", flush=True)

    # (ii) binseg_r（R 语义 + 瓶颈）Q 扫描（threshold=3ln n 与 4ln n）
    for Q in [5, 10, 20, 50]:
        for mult in [3.0, 4.0]:
            for ds, y, n, ann in datasets:
                cps = L.binseg_r(y, pen=mult * np.log(n), Q=Q)
                f1 = M.f_measure(ann, cps.tolist())
                rows.append({"family": f"binseg_r_Q{Q}_m{int(mult)}", "param": Q,
                             "dataset": ds, "k": len(cps), "f1": f1,
                             "covering": M.covering(ann, cps.tolist(), n)})
        print(f"binseg_r Q={Q} done", flush=True)

    # (iii) 教科书 binseg 阈值扫描
    for c in [2.0, 3.0, 4.0, 6.0]:
        for ds, y, n, ann in datasets:
            cps = binseg_textbook(y, beta=c * np.log(n))
            f1 = M.f_measure(ann, cps.tolist())
            rows.append({"family": "binseg_textbook", "param": c, "dataset": ds,
                         "k": len(cps), "f1": f1, "covering": M.covering(ann, cps.tolist(), n)})
        print(f"binseg_textbook c={c} done", flush=True)

    out = ROOT / "results" / "raw" / "landscape.json"
    out.write_text(json.dumps(rows))
    print("saved", out)

    # 汇总
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["family"], r["param"])].append(r["f1"])
    print("\n=== 景观（按 F1 排序）===")
    for (fam, p), v in sorted(agg.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{fam:22s} param={p:<6} F1={np.mean(v):.4f}")


if __name__ == "__main__":
    main()
