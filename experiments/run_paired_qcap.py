# -*- coding: utf-8 -*-
"""
run_paired_qcap.py — B3 配对重跑（EXP-PAIREDQ-001）

动机：表 6（Q=200，来自 run_qcap.py）与表 6b（Q=n//2，来自 run_uncapped.py）使用
**不同随机种子**，因此两者数字不可直接相减。本脚本在同一条序列上同时跑
Q=200 与 Q=n//2（含 floor / no-floor），得到**配对**的截断偏差。

种子方案与 run_uncapped.py 完全一致（seed = ci*100 + rep，ci 为 72-cell 全网格编号
从 1 起），因此 n=1000 层的 qinf 结果应逐位复现 uncapped.jsonl —— 脚本自带该自检。

输出 results/raw/paired_qcap.jsonl
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

OUT = ROOT / "results" / "raw" / "paired_qcap.jsonl"

# 与 run_uncapped.py 完全一致，用于复现 ci 编号
NS = [1000, 10000]
KS = [0, 5, 20]
NOISES = ["gauss", "t3", "ar1_09"]
AMPS = [1.0, 3.0]
SIGMAS = [1.0, 10.0]
REPS = 3
ALL_CELLS = [(n, k, nz, a, s) for n in NS for k in KS for nz in NOISES
             for a in AMPS for s in SIGMAS]

# 只跑 n=1000（n=10000 的 σ=10 / qinf 单条约 100s，留作后续；见 README）
TARGET_NS = [1000]


def binseg_floor(y, Q):
    n = len(y)
    sigma2 = L.mad_sigma(y) ** 2
    pen = max(3.0 * np.log(n), 2.0 * np.log(n) * sigma2)
    return L.binseg_r(y, pen=pen, Q=Q)


def main():
    cells = [(ci, c) for ci, c in enumerate(ALL_CELLS, 1) if c[0] in TARGET_NS]
    fout = open(OUT, "w")
    for ci, (n, k, nz, a, s) in cells:
        logn = np.log(n)
        q_inf = max(k + 1, n // 2)
        for rep in range(REPS):
            seed = ci * 100 + rep
            y, true_cps = S.gen_series(n, k, nz, a, s, seed=seed)
            for name, fn in (
                ("binseg_q200", lambda y=y: L.binseg_r(y, pen=3.0 * logn, Q=200)),
                ("binseg_q200_floor", lambda y=y: binseg_floor(y, 200)),
                ("binseg_qinf", lambda y=y: L.binseg_r(y, pen=3.0 * logn, Q=q_inf)),
                ("binseg_qinf_floor", lambda y=y: binseg_floor(y, q_inf)),
            ):
                t0 = time.perf_counter()
                cps = fn(y)
                dt = time.perf_counter() - t0
                fout.write(json.dumps({
                    "dataset": f"syn_{n}_{k}_{nz}_a{a}_s{s}_{rep}", "rep": rep,
                    "seed": seed, "algorithm": name, "k": len(cps),
                    "k_true": len(true_cps), "runtime_s": dt,
                    "n": n, "k_true_cell": k, "noise": nz, "amp": a, "sigma": s,
                }) + "\n")
                fout.flush()
        print(f"cell {ci} (n={n},k={k},{nz},a={a},s={s}) done", flush=True)
    fout.close()

    # ---- 自检：qinf 是否逐位复现 uncapped.jsonl ----
    unc = ROOT / "results" / "raw" / "uncapped.jsonl"
    if unc.exists():
        ref = {}
        for line in unc.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["n"] in TARGET_NS:
                    ref[(r["dataset"], r["algorithm"])] = r["k"]
        mine = {}
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                mine[(r["dataset"], r["algorithm"])] = r["k"]
        common = [kk for kk in mine if kk in ref]
        same = sum(1 for kk in common if mine[kk] == ref[kk])
        print(f"\n[自检] 与 uncapped.jsonl 重叠 {len(common)} 条，k 逐位一致 {same} 条"
              f"{'  ✓ 种子方案一致' if same == len(common) and common else '  ✗ 不一致！'}")

    # ---- 配对结果 ----
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    print("\n=== 配对结果：同一条序列上 Q=200 vs Q=n//2（k 误差 = k_pred − k_true）===")
    print(f"{'noise':8s} {'σ':>4s} | {'q200':>9s} {'qinf':>9s} {'低估×':>7s} | "
          f"{'q200_fl':>9s} {'qinf_fl':>9s}")
    for n in TARGET_NS:
        print(f"-- n={n} --")
        for nz in NOISES:
            for s in SIGMAS:
                g = {(r["algorithm"], r["dataset"]): r["k"] - r["k_true"]
                     for r in rows if r["noise"] == nz and r["sigma"] == s and r["n"] == n}
                keys = sorted({d for (_, d) in g})
                if not keys:
                    continue
                q200 = np.mean([g[("binseg_q200", d)] for d in keys])
                qinf = np.mean([g[("binseg_qinf", d)] for d in keys])
                q200f = np.mean([g[("binseg_q200_floor", d)] for d in keys])
                qinff = np.mean([g[("binseg_qinf_floor", d)] for d in keys])
                ratio = f"{qinf/q200:.2f}×" if q200 > 1e-9 else "n/a"
                print(f"{nz:8s} {s:4g} | {q200:+9.2f} {qinf:+9.2f} {ratio:>7s} | "
                      f"{q200f:+9.2f} {qinff:+9.2f}")


if __name__ == "__main__":
    main()
