# -*- coding: utf-8 -*-
"""run_synth.py — 合成数据网格实验。

网格：n ∈ {1000, 10000} × k ∈ {0, 5, 20} × noise ∈ {gauss, t3, ar1_09}
      × amp ∈ {1, 3} × σ ∈ {1, 10} × reps=3（seed=cell 内序号）
算法（原始尺度，不做 z-score——尺度自适应本身是被测性质）：
  mbic_raw    R-changepoint MBIC 语义（隐含 σ²=1 假设）
  bic_var     PELT β=2·ln(n)·var(y)（总方差锚定的"聪明"经典默认 = SC-PELT 地板）
  sc_nofloor  SC-PELT 纯置换校准（use_floor=False）
  sc_v3       SC-PELT 置换+BIC 地板（完整方法）
  binseg_raw  R BinSeg（Q=5, MBIC）
  zero        零基线
指标：precision/recall（margin = max(5, 1%n)）、k 误差、运行时。
输出：results/raw/synth_grid.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import synthetic as S  # noqa: E402

OUT = ROOT / "results" / "raw" / "synth_grid.json"

NS = [1000, 10000]
KS = [0, 5, 20]
NOISES = ["gauss", "t3", "ar1_09"]
AMPS = [1.0, 3.0]
SIGMAS = [1.0, 10.0]
REPS = 3


def pr_metrics(true_cps, pred_cps, n):
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


def run_cell(y, n, algos):
    out = {}
    for name, fn in algos.items():
        t0 = time.perf_counter()
        cps = fn(y)
        dt = time.perf_counter() - t0
        out[name] = {"cps": [int(c) for c in cps], "runtime_s": dt, "k": len(cps)}
    return out


def main(pilot=False):
    if pilot:
        ns, reps = [1000], 1
        cells = [(1000, 5, "gauss", 3.0, 1.0), (1000, 5, "gauss", 3.0, 10.0)]
    else:
        ns, reps = NS, REPS
        cells = None

    rows = []
    cell_iter = (
        cells
        if cells
        else [(n, k, nz, a, s) for n in ns for k in KS for nz in NOISES for a in AMPS for s in SIGMAS]
    )
    for ci, (n, k, nz, a, s) in enumerate(cell_iter):
        for rep in range(reps):
            seed = ci * 100 + rep
            y, true_cps = S.gen_series(n, k, nz, a, s, seed=seed)
            logn = np.log(n)

            algos = {
                "mbic_raw": lambda y: L.pelt_mbic(y, pen=3.0 * np.log(len(y))),
                "bic_var": lambda y: L.pelt(y, beta=2.0 * np.log(len(y)) * np.var(y)),
                "sc_nofloor": lambda y: L.sc_pelt(y, seed=seed, use_floor=False),
                "sc_v3": lambda y: L.sc_pelt(y, seed=seed, use_floor=True),
                "binseg_raw": lambda y: L.binseg_r(y, pen=3.0 * np.log(len(y)), Q=5),
                "zero": lambda y: L.zero(y),
            }
            res = run_cell(y, n, algos)
            for name, r in res.items():
                p, rec, f1 = pr_metrics(true_cps, r["cps"], n)
                rows.append({
                    "n": n, "k": k, "noise": nz, "amp": a, "sigma": s, "rep": rep,
                    "algorithm": name, "k_pred": r["k"], "k_true": len(true_cps),
                    "precision": p, "recall": rec, "f1": f1,
                    "runtime_s": r["runtime_s"], "cps": r["cps"],
                })
        print(f"cell {ci+1}/{len(cell_iter)} done: n={n} k={k} {nz} amp={a} sig={s}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    print(f"saved {len(rows)} rows -> {OUT}")

    # 汇总
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["algorithm"], r["sigma"], r["noise"])].append((r["precision"], r["recall"], r["k_pred"] - r["k_true"]))
    print("\n=== 汇总（precision / recall / k误差）===")
    for (alg, sig, nz), v in sorted(agg.items()):
        P = np.mean([x[0] for x in v]); R = np.mean([x[1] for x in v]); KE = np.mean([x[2] for x in v])
        print(f"{alg:11s} σ={sig:<5} {nz:7s}: P={P:.3f} R={R:.3f} kerr={KE:+7.2f}")


if __name__ == "__main__":
    main(pilot="--pilot" in sys.argv)
