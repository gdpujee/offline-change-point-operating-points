# -*- coding: utf-8 -*-
"""
run_qcap.py — Q 容量上限闭环实验（机理→算法链的最后一环）。

假设（来自机理研究 EVID-006）：TCPD 上 Q=5 不起作用（瓶颈先停）；
合成 k=20 时 Q=5 导致欠检测（kerr≈−3.3）；但直接放开 Q 在 σ=10 时
因 MBIC 阈值不随噪声尺度缩放（隐含 σ²=1）可能过检测。

对比（合成网格同 run_synth 因子，3 reps + TCPD）：
- binseg_q5   ：R 语义 BinSeg，Q=5（TCPDBench 默认）
- binseg_q200 ：同上，Q=200（近似无上限）
- binseg_q200_floor：Q=200 + 阈值地板 max(3ln n, 2ln n·σ̂²_MAD²)（尺度校准探索）
输出：results/raw/qcap.jsonl（增量）
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import metrics_tcpdbench as M  # noqa: E402
import synthetic as S  # noqa: E402
import tcpd_data as D  # noqa: E402

OUT = ROOT / "results" / "raw" / "qcap.jsonl"

NS = [1000, 10000]
KS = [0, 5, 20]
NOISES = ["gauss", "t3", "ar1_09"]
AMPS = [1.0, 3.0]
SIGMAS = [1.0, 10.0]
REPS = 3


def binseg_floor(y, Q):
    """瓶颈贪心 + 尺度校准阈值地板：max(3ln n, 2ln n·σ̂²)。

    瓶颈规则需要 per-split 阈值（Δ ≥ pen + ln m1 + ln m2 − ln m），
    这里把 pen 换成 max(3ln n, 2ln n·σ̂²)。binseg_r 的 pen 参数即 3ln n；
    为支持地板，直接重实现（复用 binseg_r 的候选/瓶颈逻辑，替换 pen）。
    """
    n = len(y)
    sigma2 = L.mad_sigma(y) ** 2
    pen = max(3.0 * np.log(n), 2.0 * np.log(n) * sigma2)
    return L.binseg_r(y, pen=pen, Q=Q)


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], r["rep"], r["algorithm"]))
    rows_out = open(OUT, "a")

    def emit(row):
        rows_out.write(json.dumps(row) + "\n")
        rows_out.flush()

    def pr_metrics(true_cps, pred_cps, n):
        tol = max(5, int(round(0.01 * n)))
        used = [False] * len(pred_cps)
        tp = 0
        for t in true_cps:
            best = None
            for i, g in enumerate(pred_cps):
                if not used[i] and abs(t - g) <= tol:
                    if best is None or abs(t - pred_cps[best]) > abs(t - g):
                        best = i
            if best is not None:
                used[best] = True
                tp += 1
        rec = tp / len(true_cps) if true_cps else 1.0
        prec = sum(used) / len(pred_cps) if pred_cps else 1.0
        return prec, rec

    # 合成
    cells = [(n, k, nz, a, s) for n in NS for k in KS for nz in NOISES for a in AMPS for s in SIGMAS]
    for ci, (n, k, nz, a, s) in enumerate(cells):
        for rep in range(REPS):
            seed = ci * 100 + rep
            y, true_cps = S.gen_series(n, k, nz, a, s, seed=seed)
            logn = np.log(n)
            # 续跑键必须与下面写出的 dataset **完全一致**。旧代码用 f"syn{seed}"，
            # 与写出的 f"syn_{n}_{k}_{nz}_a{a}_s{s}_{rep}" 永不相等，导致 done 从不命中、
            # 每次重跑都把合成行重复追加（qcap.jsonl 771 -> 1419 行），
            # 进而使 make_tables.py 的 108/108、18/18 计数变成 216/216、36/36。
            dsname = f"syn_{n}_{k}_{nz}_a{a}_s{s}_{rep}"
            algos = {
                "binseg_q5": lambda y: L.binseg_r(y, pen=3.0 * logn, Q=5),
                "binseg_q200": lambda y: L.binseg_r(y, pen=3.0 * logn, Q=200),
                "binseg_q200_floor": lambda y: binseg_floor(y, 200),
            }
            for name, fn in algos.items():
                if (dsname, rep, name) in done:
                    continue
                t0 = time.perf_counter()
                cps = fn(y)
                dt = time.perf_counter() - t0
                p, r = pr_metrics(true_cps, list(map(int, cps)), n)
                emit({"dataset": dsname, "rep": rep,
                      "algorithm": name, "k": len(cps), "k_true": len(true_cps),
                      "precision": p, "recall": r, "runtime_s": dt,
                      "n": n, "k_true_cell": k, "noise": nz, "sigma": s})
        print(f"cell {ci+1}/{len(cells)} done", flush=True)

    # TCPD
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except FileNotFoundError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        ann = D.load_annotations(ds)
        logn = np.log(n)
        algos = {
            "binseg_q5": lambda: L.binseg_r(y, pen=3.0 * logn, Q=5),
            "binseg_q200": lambda: L.binseg_r(y, pen=3.0 * logn, Q=200),
            "binseg_q200_floor": lambda: binseg_floor(y, 200),
        }
        for name, fn in algos.items():
            if (ds, 0, name) in done:
                continue
            cps = fn()
            f1, p, r = M.f_measure(ann, cps.tolist(), return_PR=True)
            cov = M.covering(ann, cps.tolist(), n)
            emit({"dataset": ds, "rep": 0, "algorithm": name, "k": len(cps),
                  "f1": f1, "precision": p, "recall": r, "covering": cov})
        print(f"tcpd {ds} done", flush=True)
    rows_out.close()

    # 汇总
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    from collections import defaultdict
    syn = defaultdict(list)
    tcp = defaultdict(list)
    for r in rows:
        if r["dataset"].startswith("syn_"):
            syn[r["algorithm"]].append(r)
        else:
            tcp[r["algorithm"]].append(r)
    print("\n=== 合成（按 noise×σ）===")
    agg = defaultdict(list)
    for _a, rs in syn.items():
        for r in rs:
            agg[(r["algorithm"], r["noise"], r["sigma"])].append(r)
    for (a, nz, s), v in sorted(agg.items()):
        print(f"{a:20s} {nz:7s} σ={s:<5}: P={np.mean([x['precision'] for x in v]):.3f} "
              f"R={np.mean([x['recall'] for x in v]):.3f} "
              f"kerr={np.mean([x['k']-x['k_true'] for x in v]):+8.2f}")
    print("\n=== TCPD ===")
    for a, v in tcp.items():
        print(f"{a:20s} F1={np.mean([x['f1'] for x in v]):.4f} "
              f"Cov={np.mean([x['covering'] for x in v]):.4f} k={np.mean([x['k'] for x in v]):.2f}")


if __name__ == "__main__":
    main()
