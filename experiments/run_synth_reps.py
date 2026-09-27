# -*- coding: utf-8 -*-
"""run_synth_reps.py — 加大重复次数的合成网格（外部评审第 5 条）。

为什么有这个文件
----------------
旧的 `run_synth.py` 每格只有 3 次重复，报的是**点估计**，没有离散度。
评审要求：每格 ≥50 次，报 mean / median / SE / bootstrap CI，并把 AR(1) 的
φ 扫开（而不是只在 φ=0.9 采一个点）。

本脚本产出**逐条** raw 记录（每条 = 一个序列 × 一个算法），离散度统计留给
`analyze_synth_reps.py` 去做 —— raw 与分析分离，跟全库约定一致。

网格
----
A. 主网格（尺度敏感性 / UBG-floor 修复的主证据）
   n ∈ {500, 1000, 5000, 10000} × k ∈ {0,5,20}
   × noise ∈ {gauss, t3, ar1_09} × amp ∈ {1,3} × σ ∈ {1,10}
   = 144 格 × 50 次 = 7200 条序列

B. AR(1) φ 扫描（Proposition 2 / Corollary 2 的经验对应物）
   n ∈ {1000, 10000} × k = 5 × φ ∈ {0, 0.2, 0.5, 0.7, 0.9}
   × amp = 3 × σ = 10 = 10 格 × 50 次 = 500 条序列

算法与 `run_synth.py` 完全一致（6 个），指标口径也一致（margin = max(5, 1%·n)）。

输出：results/raw/synth_reps.jsonl（overwrite；每次重跑整体重写，逐条确定性）

用法：
    python experiments/run_synth_reps.py            # 全量
    python experiments/run_synth_reps.py --pilot    # 每格 2 次，冒烟用
    python experiments/run_synth_reps.py --jobs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import synthetic as S  # noqa: E402

# R54: env-overridable, so a 100-repetition run can be written to a SIDE file
# and compared before anything overwrites the frozen 50-repetition raw output.
import os  # noqa: E402
OUT = Path(os.environ.get("TCPD_SYNTH_OUT",
                          str(ROOT / "results" / "raw" / "synth_reps.jsonl")))

REPS = 50

# (n, k, noise, amp, sigma)
GRID_MAIN = [
    (n, k, nz, a, s)
    for n in (500, 1000, 5000, 10000)
    for k in (0, 5, 20)
    for nz in ("gauss", "t3", "ar1_09")
    for a in (1.0, 3.0)
    for s in (1.0, 10.0)
]

GRID_PHI = [
    (n, 5, f"ar1_{dd:02d}", 3.0, 10.0)
    for n in (1000, 10000)
    for dd in (0, 2, 5, 7, 9)
]


def pr_metrics(true_cps, pred_cps, n):
    """与 run_synth.py 逐字一致：margin = max(5, 1%·n)，双向贪心匹配。"""
    tol = max(5, int(round(0.01 * n)))
    used = [False] * len(pred_cps)
    tp_t = 0
    for t in true_cps:
        best = None
        for i, g in enumerate(pred_cps):
            if not used[i] and abs(t - g) <= tol:
                if best is None or abs(t - pred_cps[best]) > abs(t - g):
                    best = i
        if best is not None:
            used[best] = True
            tp_t += 1
    recall = tp_t / len(true_cps) if true_cps else 1.0
    precision = sum(used) / len(pred_cps) if pred_cps else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def algos_for(seed):
    return {
        "mbic_raw": lambda y: L.pelt_mbic(y, pen=3.0 * np.log(len(y))),
        "bic_var": lambda y: L.pelt(y, beta=2.0 * np.log(len(y)) * np.var(y)),
        "sc_nofloor": lambda y: L.sc_pelt(y, seed=seed, use_floor=False),
        "sc_v3": lambda y: L.sc_pelt(y, seed=seed, use_floor=True),
        "binseg_raw": lambda y: L.binseg_r(y, pen=3.0 * np.log(len(y)), Q=5),
        "zero": lambda y: L.zero(y),
    }


def run_one(task):
    """跑一格。返回该格所有 (序列 × 算法) 的记录。"""
    grid, cell_idx, (n, k, nz, a, s), reps = task
    rows = []
    for rep in range(reps):
        seed = cell_idx * 1000 + rep
        y, true_cps = S.gen_series(n, k, nz, a, s, seed=seed)
        algos = algos_for(seed)
        for name, fn in algos.items():
            t0 = time.perf_counter()
            cps = [int(c) for c in fn(y)]
            dt = time.perf_counter() - t0
            p, rec, f1 = pr_metrics(true_cps, cps, n)
            rows.append({
                "grid": grid,
                "n": n, "k": k, "noise": nz, "amp": a, "sigma": s,
                "rep": rep, "seed": seed, "algorithm": name,
                "k_pred": len(cps), "k_true": len(true_cps),
                "k_err": len(cps) - len(true_cps),
                "precision": p, "recall": rec, "f1": f1,
                "runtime_s": dt, "cps": cps,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="每格 2 次，冒烟用")
    ap.add_argument("--jobs", type=int, default=10)
    # seed = cell_idx*1000 + rep, so the first 50 reps of a --reps 100 run are
    # BIT-IDENTICAL to a --reps 50 run: that is the cross-check used in R54.
    ap.add_argument("--reps", type=int, default=REPS,
                    help="每格重复次数（默认 50；种子只与 rep 有关，故可前缀对拍）")
    args = ap.parse_args()

    reps = 2 if args.pilot else args.reps
    tasks = []
    for grid, cells in (("main", GRID_MAIN), ("phi", GRID_PHI)):
        for ci, cell in enumerate(cells):
            tasks.append((grid, ci, cell, reps))

    print(f"cells={len(tasks)} reps={reps} jobs={args.jobs} "
          f"=> ~{len(tasks)*reps} series", flush=True)
    print(f"out={OUT}", flush=True)

    t0 = time.time()
    all_rows = []
    done = 0
    with Pool(args.jobs) as pool:
        for rows in pool.imap_unordered(run_one, tasks, chunksize=1):
            all_rows.extend(rows)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} cells  ({time.time()-t0:.0f}s)", flush=True)

    # 确定性排序（多进程完成顺序不定）
    all_rows.sort(key=lambda r: (r["grid"], r["n"], r["k"], r["noise"],
                                 r["amp"], r["sigma"], r["rep"], r["algorithm"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"saved {len(all_rows)} rows -> {OUT}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
