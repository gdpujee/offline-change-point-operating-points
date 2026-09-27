# -*- coding: utf-8 -*-
"""
run_extra_baselines.py — 补充基线与中等自相关实验（内部评审 Round 1 行动项 A3/A7）。

A3: Bottom-Up 分割（教科书语义：单点起步，反复合并相邻段中「分裂增益」最小者，
    直到最小增益 < 阈值）。阈值 β = 3ln n（MBIC 一致口径，z-scored）。
A7: ar1_05（φ=0.5 中等自相关）cell：检验 σ̂ 的 AR 校正在中等自相关下是否有效。
输出：results/raw/extra_baselines.jsonl（增量）
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

OUT = ROOT / "results" / "raw" / "extra_baselines.jsonl"


def bottomup(y: np.ndarray, beta: float) -> np.ndarray:
    """教科书 Bottom-Up：单点段起步，合并相邻段中分裂增益最小者，直到 < beta。"""
    ss = L.SumStats(y)
    n = ss.n
    if n < 2:
        return np.array([], dtype=int)
    # boundaries: 段起点列表；段 i = [b[i], b[i+1])
    b = list(range(n + 1))
    # gain[i] = 合并段 i 与 i+1 的损失 = 分裂增益（在边界 b[i+1] 处）
    def gain(i):
        a, s, e = b[i], b[i + 1], b[i + 2]
        return ss.sse(a, e) - ss.sse(a, s) - ss.sse(s, e)
    g = [gain(i) for i in range(len(b) - 2)]
    while len(b) > 2:
        i = int(np.argmin(g))
        if g[i] >= beta:
            break  # 最不显著的相邻界都显著 → 停止
        # 合并段 i 与 i+1：删除边界 b[i+1]
        del b[i + 1]
        del g[i]
        if i > 0:
            g[i - 1] = gain(i - 1)
        if i < len(g):
            g[i] = gain(i)
    return np.array(b[1:-1], dtype=int) - 1  # 内部边界 → 0-based CP（左段末点）


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                # 必须带上 rep：否则一次中断的重跑会把剩余 rep 全部跳过（静默丢数据）。
                # TCPD 行没有 rep 字段，故为 None。
                done.add((r.get("dataset"), r.get("rep"), r.get("algorithm")))
    f = open(OUT, "a")

    def emit(row):
        f.write(json.dumps(row) + "\n")
        f.flush()

    # A3: TCPD 上的 bottomup
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except FileNotFoundError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        ann = D.load_annotations(ds)
        if (ds, None, "bottomup") in done:
            continue
        t0 = time.perf_counter()
        cps = bottomup(y, 3.0 * np.log(n))
        dt = time.perf_counter() - t0
        f1, p, r = M.f_measure(ann, cps.tolist(), return_PR=True)
        cov = M.covering(ann, cps.tolist(), n)
        emit({"dataset": ds, "algorithm": "bottomup", "k": len(cps), "cps": cps.tolist(),
              "f1": f1, "precision": p, "recall": r, "covering": cov, "runtime_s": dt})
        print(f"tcpd {ds} bottomup k={len(cps)} f1={f1:.3f}", flush=True)

    # A7: ar1_05 中等自相关 cell（n=5000, k=5, amp=3）
    def pr(true_cps, pred_cps, n_):
        tol = max(5, int(round(0.01 * n_)))
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
        return (sum(used) / len(pred_cps) if pred_cps else 1.0,
                tp / len(true_cps) if true_cps else 1.0)

    for s in [1.0, 10.0]:
        for rep in range(3):
            seed = 777 + rep
            y, true_cps = S.gen_series(5000, 5, "ar1_05", 3.0, s, seed=seed)
            logn = np.log(5000)
            sigma2 = L.mad_sigma(y) ** 2
            algos = {
                "binseg_q5": lambda y: L.binseg_r(y, pen=3.0 * logn, Q=5),
                "binseg_q200_floor": lambda y: L.binseg_r(
                    y, pen=max(3.0 * logn, 2.0 * logn * sigma2), Q=200),
                "sc_v3": lambda y: L.sc_pelt(y, seed=seed, use_floor=True),
            }
            # 同 run_qcap.py：续跑键必须与写出的 dataset 一致。
            dsname = f"syn_ar1_05_s{s}"
            for name, fn in algos.items():
                if (dsname, rep, name) in done:
                    continue
                cps = fn(y)
                p, r = pr(true_cps, list(map(int, cps)), 5000)
                emit({"dataset": dsname, "rep": rep, "algorithm": name,
                      "k": len(cps), "k_true": 5, "precision": p, "recall": r,
                      "n": 5000, "noise": "ar1_05", "sigma": s})
            print(f"ar1_05 s={s} rep={rep} done", flush=True)
    f.close()

    # 汇总
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    tcp = [r for r in rows if not r.get("dataset", "").startswith("syn_")]
    syn = [r for r in rows if r.get("dataset", "").startswith("syn_")]
    if tcp:
        print(f"\n=== TCPD bottomup: F1={np.mean([r['f1'] for r in tcp]):.4f} "
              f"Cov={np.mean([r['covering'] for r in tcp]):.4f} k={np.mean([r['k'] for r in tcp]):.2f} "
              f"time={np.mean([r['runtime_s'] for r in tcp]):.4f}s ===")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in syn:
        agg[(r["algorithm"], r["sigma"])].append(r)
    print("=== ar1_05 (φ=0.5, n=5000, k=5, amp=3σ) ===")
    for (a, s), v in sorted(agg.items()):
        print(f"{a:20s} σ={s:<5}: P={np.mean([x['precision'] for x in v]):.3f} "
              f"R={np.mean([x['recall'] for x in v]):.3f} "
              f"kerr={np.mean([x['k']-x['k_true'] for x in v]):+.2f}")


if __name__ == "__main__":
    main()
