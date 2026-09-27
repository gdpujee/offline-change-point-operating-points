# -*- coding: utf-8 -*-
"""
phantom_impact.py — 幻影变点双口径对照。

协议 A（published）：按 TCPDBench 实际导出口径——PELT/BinSeg/SegNeigh 末尾附加
n−1 幻影；AMOC 无变点时输出 [n−1]。
协议 B（corrected）：幻影/哨兵剔除（即 tcpd_main.json 的口径）。

目的：① 验证协议 A 能复现论文 published 数字（闭合复现回路）；
     ② 量化幻影对 F1/Covering 的扭曲。
输出：results/raw/phantom_impact.json
      results/processed/phantom_impact_{all41,paper32}.csv   （每个子集一个文件）

注：`results/processed/phantom_impact.csv` 是一个**遗留孤儿产物**——本脚本从不写它
（写入路径见 main() 末尾的 `phantom_impact_{subset_name}.csv`）。全仓无任何代码读它，
它也不被论文引用。属"低危未处置"，记录在此以免被误当作本脚本的输出。
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

OUT = ROOT / "results" / "raw" / "phantom_impact.json"


def main():
    rows = []
    for ds in D.BENCH_DATASETS:
        try:
            y, n_obs = D.load_series(ds)
        except FileNotFoundError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        ann = D.load_annotations(ds)
        logn = np.log(n_obs)
        algos = {
            "amoc": L.amoc(y, pen=3.0 * logn),
            "binseg": L.binseg_r(y, pen=3.0 * logn, Q=5),
            "pelt": L.pelt_mbic(y, pen=3.0 * logn),
            "segneigh": L.segneigh(y, beta=2.0 * logn, Q=5),
            "zero": L.zero(y),
        }
        for name, cps in algos.items():
            cps_l = cps.tolist()
            # 协议 A：published 口径
            if name in ("pelt", "binseg", "segneigh"):
                cpsA = cps_l + [n_obs - 1]
            elif name == "amoc":
                cpsA = cps_l if cps_l else [n_obs - 1]
            else:
                cpsA = cps_l
            f1A, pA, rA = M.f_measure(ann, cpsA, return_PR=True)
            covA = M.covering(ann, cpsA, n_obs)
            # 协议 B：corrected 口径
            f1B, pB, rB = M.f_measure(ann, cps_l, return_PR=True)
            covB = M.covering(ann, cps_l, n_obs)
            rows.append({"dataset": ds, "algorithm": name,
                         "f1_published": f1A, "covering_published": covA,
                         "f1_corrected": f1B, "covering_corrected": covB,
                         "precision_published": pA, "recall_published": rA,
                         "k": len(cps_l)})
        print(ds, "done", flush=True)

    OUT.write_text(json.dumps(rows))
    print("saved", OUT)

    # 汇总：论文精确子集（univariate 非 QC，32 条）为主口径，全部 41 条为参考
    MULTI = {"apple", "bee_waggle_6", "occupancy", "run_log"}  # TCPDBench MULTIDATASETS
    QC = {f"quality_control_{i}" for i in range(1, 6)}          # TCPDBench QC_DATASETS
    from collections import defaultdict
    for subset_name, pred in [("paper32", lambda ds: ds not in MULTI and ds not in QC),
                              ("all41", lambda ds: True)]:
        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if not pred(r["dataset"]):
                continue
            for f in ["f1_published", "f1_corrected", "covering_published", "covering_corrected"]:
                agg[r["algorithm"]][f].append(r[f])
        print(f"\n=== subset: {subset_name} (n_datasets={len(agg['binseg']['f1_published'])}) ===")
        print(f"{'algorithm':10s} {'F1(pub)':>8s} {'F1(corr)':>9s} {'ΔF1':>7s} {'Cov(pub)':>9s} {'Cov(corr)':>10s}")
        csv_rows = []
        for a in ["amoc", "binseg", "pelt", "segneigh", "zero"]:
            m = {f: float(np.mean(v)) for f, v in agg[a].items()}
            print(f"{a:10s} {m['f1_published']:8.4f} {m['f1_corrected']:9.4f} "
                  f"{m['f1_corrected']-m['f1_published']:+7.4f} "
                  f"{m['covering_published']:9.4f} {m['covering_corrected']:10.4f}")
            csv_rows.append([subset_name, a, m["f1_published"], m["f1_corrected"],
                             m["covering_published"], m["covering_corrected"]])
        csv = ROOT / "results" / "processed" / f"phantom_impact_{subset_name}.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        with open(csv, "w") as f:
            f.write("subset,algorithm,f1_published,f1_corrected,covering_published,covering_corrected\n")
            for r in csv_rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print("saved", csv)


if __name__ == "__main__":
    main()
