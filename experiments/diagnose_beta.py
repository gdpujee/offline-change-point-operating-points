# -*- coding: utf-8 -*-
"""diagnose_beta.py — SC-PELT 校准诊断：β = c·max(S) 扫描 + Δ 背景结构检查。

回答一个问题：surrogate null 与真实数据的差距是「差一个尺度系数」还是
「形状/结构根本不同」（趋势、季节、异方差造成的 Δ 背景抬升）？
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


def surrogate_stats(y, B=100, beta_pilot=0.5, seed=12345, rescale=False):
    rng = np.random.default_rng(seed)
    n = len(y)
    ss = L.SumStats(y)
    sigma = L.mad_sigma(y)
    segs = L.pelt(y, beta=beta_pilot, minseglen=2)
    bounds = np.concatenate(([0], segs + 1, [n]))
    r = y.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            r[a:b] -= y[a:b].mean()
    if rescale:
        sd_r = float(np.std(r))
        if sd_r > 0:
            r = r * (sigma / sd_r)
    L_ = int(max(8, min(n // 10, round(2 * n ** (1 / 3)))))
    S = np.empty(B)
    for b in range(B):
        yb = L.block_permute(r, L_, rng)
        S[b] = L.max_delta(yb)
    return S


def main():
    rows = []
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except FileNotFoundError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        ann = D.load_annotations(ds)
        S = surrogate_stats(y, rescale=False)
        smax = float(np.max(S))
        for c in [1, 1.5, 2, 3, 5, 8]:
            beta = c * smax
            cps = L.pelt(y, beta=beta)
            f1, p, r = M.f_measure(ann, cps.tolist(), return_PR=True)
            rows.append({"dataset": ds, "c": c, "beta": beta, "k": len(cps),
                         "f1": f1, "precision": p, "recall": r})
        # 诊断：真实 Δ 背景的 99% 分位 vs surrogate max
        ss = L.SumStats(y)
        c_arr = np.arange(0, n - 1)
        d = ss.sse(0, n) - ss.sse_vec(0, c_arr + 1) - ss.sse_vec(c_arr + 1, n)
        rows.append({"dataset": ds, "c": "diag", "beta": float(np.quantile(d, 0.99)),
                     "k": 0, "f1": smax, "precision": float(np.quantile(d, 0.999)),
                     "recall": float(np.median(d))})
        print(f"{ds:22s} smax={smax:8.3f} q99(Δ)={np.quantile(d,0.99):8.3f} "
              f"med(Δ)={np.median(d):7.3f}", flush=True)

    out = ROOT / "results" / "raw" / "diagnose_beta.json"
    out.write_text(json.dumps(rows))
    print("saved", out)

    # 汇总
    print("\n=== c 扫描平均（37 数据集）===")
    for c in [1, 1.5, 2, 3, 5, 8]:
        sel = [r for r in rows if r["c"] == c]
        print(f"c={c:4}: F1={np.mean([r['f1'] for r in sel]):.4f} "
              f"k={np.mean([r['k'] for r in sel]):7.2f} "
              f"prec={np.mean([r['precision'] for r in sel]):.4f}")


if __name__ == "__main__":
    main()
