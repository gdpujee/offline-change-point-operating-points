# -*- coding: utf-8 -*-
"""analyze_synth_reps.py — 给重复合成网格加离散度（R58 起默认 100 次）。

读 `results/raw/synth_reps.jsonl`（run_synth_reps.py 产出，逐条 raw），输出每格的
mean / median / SE / bootstrap 95% CI —— 评审要求的最小离散度报告。

另外把 AR(1) 的 φ 扫开，量 Proposition 2 的经验对应物：
    σ̂/σ  的理论值 = √(1−φ)          （Proposition 2：E[σ̂] = σ√(1−φ)）
    实测值 = mean(mad_sigma(y)) / σ， 50 次重复
EVID-036 原来只有 6 次重复（且只在 φ∈{0.5,0.9} 两点），这里补到多次 × 5 个 φ。

R58: TCPD_PHI_REPS 的默认值从 "50" 改为 "100"。此前默认 50 而随包提交的
synth_reps_summary.json 是 100 次——附录 B 印的那条命令因此**复现不出**已提交
的产物，差异藏在一个没写进附录的环境变量后面。种子只与 rep 有关，所以 100 次的
前 50 条与 50 次的逐位相同，设 TCPD_PHI_REPS=50 仍可复现早先那一版。

输出：
- results/raw/synth_reps_summary.json   （逐格统计量 + φ 扫描）
- results/raw/sigmahat_phi_reps.json    （φ 扫描的逐次原始值）
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import synthetic as S  # noqa: E402

# R54: paths overrideable so a 100-repetition run can be analysed next to the
# frozen 50-repetition one instead of over it. The phi sweep is regenerated here
# (seed = 900000 + n*100 + dd*10 + rep), so PHI_REPS=100 reproduces the first 50
# as a prefix -- the same cross-check as for the main grid.
import os  # noqa: E402

SRC = Path(os.environ.get("TCPD_SYNTH_IN",
                          str(ROOT / "results" / "raw" / "synth_reps.jsonl")))
OUT = Path(os.environ.get("TCPD_SYNTH_SUMMARY",
                          str(ROOT / "results" / "raw" / "synth_reps_summary.json")))
OUT_PHI_RAW = Path(os.environ.get("TCPD_PHI_RAW",
                                  str(ROOT / "results" / "raw" / "sigmahat_phi_reps.json")))

B = 10000          # bootstrap 重抽样次数
PHIS = (0.0, 0.2, 0.5, 0.7, 0.9)
PHI_NS = (1000, 10000)
PHI_REPS = int(os.environ.get("TCPD_PHI_REPS", "100"))
PHI_SIGMA = 10.0


def boot_ci(xs, b=B, seed=0):
    """均值的 percentile bootstrap 95% CI。"""
    xs = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(b, n))
    means = xs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def describe(xs, seed=0):
    xs = np.asarray(xs, dtype=float)
    se = float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
    lo, hi = boot_ci(xs, seed=seed)
    return {
        "n": int(len(xs)),
        "mean": float(xs.mean()),
        "median": float(np.median(xs)),
        "sd": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
        "se": se,
        "ci95_lo": lo,
        "ci95_hi": hi,
    }


def main():
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8")]
    print("rows:", len(rows))

    buckets = defaultdict(list)
    for r in rows:
        key = (r["grid"], r["n"], r["k"], r["noise"], r["amp"], r["sigma"], r["algorithm"])
        buckets[key].append(r)

    cells = []
    for i, (key, rs) in enumerate(sorted(buckets.items())):
        grid, n, k, nz, amp, sig, alg = key
        kerr = [r["k_err"] for r in rs]
        f1 = [r["f1"] for r in rs]
        prec = [r["precision"] for r in rs]
        rec = [r["recall"] for r in rs]
        cells.append({
            "grid": grid, "n": n, "k": k, "noise": nz, "amp": amp,
            "sigma": sig, "algorithm": alg,
            "k_err": describe(kerr, seed=i),
            "f1": describe(f1, seed=i),
            "precision": describe(prec, seed=i),
            "recall": describe(rec, seed=i),
        })

    # ---- AR(1) φ 扫描：σ̂/σ 的实测 vs 理论 √(1−φ) ----
    phi_raw = []
    phi_summary = []
    for n in PHI_NS:
        for phi in PHIS:
            dd = int(round(phi * 10))
            ratios = []
            for rep in range(PHI_REPS):
                seed = 900000 + n * 100 + dd * 10 + rep
                y, _ = S.gen_series(n, 5, f"ar1_{dd:02d}", 3.0, PHI_SIGMA, seed=seed)
                ratios.append(float(L.mad_sigma(y) / PHI_SIGMA))
                phi_raw.append({"n": n, "phi": phi, "rep": rep, "seed": seed,
                                "sigmahat_over_sigma": ratios[-1]})
            d = describe(np.asarray(ratios), seed=n + dd)
            pred = float(np.sqrt(1 - phi))
            phi_summary.append({
                "n": n, "phi": phi, "reps": PHI_REPS,
                "ratio_pred": pred,
                "ratio_meas": d,
                "abs_gap": abs(d["mean"] - pred),
                "rel_gap": abs(d["mean"] - pred) / pred if pred > 0 else float("nan"),
                "pred_within_ci": bool(d["ci95_lo"] <= pred <= d["ci95_hi"]),
            })

    out = {
        "source": str(SRC.relative_to(ROOT)),
        "rows": len(rows),
        "bootstrap_B": B,
        "cells": cells,
        "phi_sweep": phi_summary,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    OUT_PHI_RAW.write_text(json.dumps(phi_raw, indent=1), encoding="utf-8")

    print(f"saved {len(cells)} cell summaries -> {OUT}")
    print(f"saved {len(phi_raw)} phi-sweep raw values -> {OUT_PHI_RAW}")
    print("\n=== AR(1) φ 扫描：σ̂/σ  vs  理论 √(1−φ) ===")
    print(f"{'n':>6} {'φ':>5} {'pred':>8} {'meas':>8} {'SE':>8} {'95% CI':>20} {'pred in CI':>10}")
    for r in phi_summary:
        d = r["ratio_meas"]
        print(f"{r['n']:6d} {r['phi']:5.1f} {r['ratio_pred']:8.4f} {d['mean']:8.4f} "
              f"{d['se']:8.4f} [{d['ci95_lo']:.4f}, {d['ci95_hi']:.4f}] "
              f"{str(r['pred_within_ci']):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
