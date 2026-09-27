# -*- coding: utf-8 -*-
"""
run_ar_fix.py — 「既然 σ̂ = σ√(1−φ)，为什么不除回去？」的直接回答。

背景：EVID-036 给出闭式 σ̂ = σ√(1−φ)。这立刻引出一个显然的修法：
    σ̂_corrected = σ̂ / √(1 − φ̂)，  φ̂ = 水平序列的滞后-1 自相关
本脚本实测这个修法，并回答"为什么论文没有采用它"。

预期（且已被证实的）困难：**φ̂ 本身被变点污染**。水平序列里的跳变在滞后-1
自相关上表现为"持续性"，因此 φ̂ 混淆了两种完全不同的来源：
    (a) 真实的序列相关（AR）；
    (b) 待检测的信号本身（变点跳变）。
在 gauss（白噪声 + 变点）上 φ̂ 应接近 0，实测却约 0.45 —— 这个偏差会把弱相关情形
过惩罚，把 k 误差从 −2.56 推到 −5.33。故"除以 √(1−φ̂)"不是免费的。

三种口径：
    floor  —— 论文采用的 β = max(3ln n, 2ln n·σ̂²)
    arfix  —— β = max(3ln n, 2ln n·(σ̂/√(1−φ̂))²)，φ̂ = clip(corr(y[:-1],y[1:]), 0, 0.99)
    oracle —— β = max(3ln n, 2ln n·(σ̂/√(1−φ_true))²)，φ_true 已知（仅作上界参照）

另按 k_true ∈ {0,5,20} 分层记录 φ̂，用来直接展示"φ̂ 随真实变点数上升而虚高"。

种子：必须与 run_uncapped.py 完全一致 —— cell 编号 ci 由**完整**网格枚举决定，
因此这里必须先枚举完整网格再筛选，否则 seed 会错位（踩过的坑）。

输出：results/raw/arfix.jsonl（逐条追加）
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

OUT = ROOT / "results" / "raw" / "arfix.jsonl"

# 与 run_uncapped.py 完全一致的网格（决定 ci → seed），不可删减后再枚举
NS = [1000, 10000]
KS = [0, 5, 20]
NOISES = ["gauss", "t3", "ar1_09"]
AMPS = [1.0, 3.0]
SIGMAS = [1.0, 10.0]
REPS = 3

PHI_TRUE = {"gauss": 0.0, "t3": 0.0, "ar1_09": 0.9}

# 本实验只跑 n=10³（ar1_09 σ=10 在 n=10⁴ 上单条约 20 s，代价主要在 floor 口径）
RUN_NS = [1000]


def phi_hat(y):
    c = np.corrcoef(y[:-1], y[1:])[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(min(max(c, 0.0), 0.99))


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["n"], r["k_true_cell"], r["noise"], r["amp"], r["sigma"],
                          r["rep"], r["mode"]))
    rows_out = open(OUT, "a")

    def emit(row):
        rows_out.write(json.dumps(row) + "\n")
        rows_out.flush()

    full = [(n, k, nz, a, s) for n in NS for k in KS for nz in NOISES
            for a in AMPS for s in SIGMAS]
    for ci, (n, k, nz, a, s) in enumerate(full, 1):
        if n not in RUN_NS:
            continue
        for rep in range(REPS):
            y, true_cps = S.gen_series(n, k, nz, a, s, seed=ci * 100 + rep)
            kt = len(true_cps)
            logn = np.log(n)
            q = max(k + 1, n // 2)
            sh = float(L.mad_sigma(y))
            ph = phi_hat(y)
            phi_t = PHI_TRUE[nz]
            variants = {
                "floor": sh ** 2,
                "arfix": (sh / np.sqrt(1.0 - ph)) ** 2,
                "oracle": (sh / np.sqrt(1.0 - phi_t)) ** 2,
            }
            for mode, sig2 in variants.items():
                if (n, k, nz, a, s, rep, mode) in done:
                    continue
                pen = max(3.0 * logn, 2.0 * logn * sig2)
                t0 = time.perf_counter()
                cps = L.binseg_r(y, pen=pen, Q=q)
                dt = time.perf_counter() - t0
                emit({"n": n, "k_true_cell": k, "noise": nz, "amp": a, "sigma": s,
                      "rep": rep, "seed": ci * 100 + rep, "mode": mode,
                      "k_true": kt, "k_pred": int(len(cps)),
                      "sigmahat": sh, "phihat": ph, "phi_true": phi_t,
                      "pen": float(pen), "runtime_s": dt})
        print(f"cell {ci}/{len(full)} done", flush=True)
    rows_out.close()

    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]

    print("\n=== k 误差（n=10³, Q=n//2, 每格 18 条）===")
    print(f"{'noise':8s} {'σ':>5s} | {'floor':>9s} {'arfix':>9s} {'oracle':>9s} | {'mean φ̂':>8s}")
    for nz in NOISES:
        for s in SIGMAS:
            def m(mode):
                sel = [r for r in rows if r["noise"] == nz and r["sigma"] == s and r["mode"] == mode]
                return np.mean([r["k_pred"] - r["k_true"] for r in sel]) if sel else float("nan")
            ph = np.mean([r["phihat"] for r in rows
                          if r["noise"] == nz and r["sigma"] == s and r["mode"] == "floor"])
            print(f"{nz:8s} {s:5g} | {m('floor'):+9.2f} {m('arfix'):+9.2f} {m('oracle'):+9.2f} | {ph:8.3f}")

    print("\n=== φ̂ 按真实变点数分层（污染的直接证据）===")
    print(f"{'noise':8s} {'k_true':>7s} {'mean φ̂':>9s} {'φ_true':>8s} {'φ̂−φ_true':>10s}")
    for nz in NOISES:
        for kt in KS:
            sel = [r for r in rows if r["noise"] == nz and r["k_true_cell"] == kt and r["mode"] == "floor"]
            if not sel:
                continue
            ph = np.mean([r["phihat"] for r in sel])
            pt = PHI_TRUE[nz]
            print(f"{nz:8s} {kt:7d} {ph:9.3f} {pt:8.2f} {ph - pt:+10.3f}")

    print("\n结论提示：若 gauss 上 φ̂ 明显 > 0，则 φ̂ 被变点污染，"
          "σ̂/√(1−φ̂) 会把弱相关情形过惩罚——这就是论文不采用该修法的理由。")


if __name__ == "__main__":
    main()
