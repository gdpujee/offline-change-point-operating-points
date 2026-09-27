# -*- coding: utf-8 -*-
"""
diagnose_mismatch.py — Δ 背景失配程度的重新量化（EXP-MISMATCH-001）。

替代已废弃的 diagnose_beta.py 的 surrogate 部分。旧版的两个缺陷（review_round3 B4/B5）：
  (1) 用 β_pilot=0.5 构造 surrogate —— 正是 FAILURES F1/F2 判定为"段均值吃掉噪声、
      null 尺度系统性偏小"的坏 pilot，导致比值分母被污染（旧的 ~500×/1290× 不可用）。
  (2) mad_sigma（一阶差分 MAD）在平滑趋势序列上结构性塌陷为 0（差分近似恒定 → MAD≈0
      → σ̂≈0 → pilot 罚参≈0 → 过分割到 minseglen → 残差恒 0 → null=0）。
      实测 fig5_data.json 中 us_population 的 B=100 个 surrogate max 全为 0.0000。

本脚本用两个**非退化**的 null：
  A) 参数化 null：iid N(0, var(y))，长度 n，B 次。这是 MBIC/BIC 罚参**实际假设**的模型
     （数据经 R scale() 标准化后 var(y)=1），因而是最直接、假设最少的对照。
  B) 块置换 null：pilot 罚参改为**总方差锚定** β_pilot = 2·ln n·var(y)（等价于 BIC 尺度，
     对标准化数据恒为 2ln n ≈ 13，绝不退化），段长下限 4；残差去均值后循环块置换。

被解释量（与旧版一致，保证可比）：真实序列 Δ 剖面的**中位数**（背景水平）
对比 null 的**最大值**（整个噪声分布的极值）。

输出：results/raw/mismatch.json + results/raw/fig5_data.json（供 make_figures 重画）
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import tcpd_data as D  # noqa: E402

OUT = ROOT / "results" / "raw" / "mismatch.json"
OUT_FIG5 = ROOT / "results" / "raw" / "fig5_data.json"
B = 200
SEED = 20260913


def delta_profile(y, ss=None):
    ss = ss or L.SumStats(y)
    n = ss.n
    c = np.arange(0, n - 1)
    return ss.sse(0, n) - ss.sse_vec(0, c + 1) - ss.sse_vec(c + 1, n)


def parametric_null(n, var_y, B, rng):
    """iid N(0, var_y) 序列的 max Δ 分布（参数化 null）。"""
    sd = float(np.sqrt(var_y))
    S = np.empty(B)
    for b in range(B):
        yb = rng.normal(0.0, sd, size=n)
        S[b] = L.max_delta(yb)
    return S


def permutation_null(y, B, rng, minseglen=4):
    """总方差锚定 pilot（β=2 ln n·var(y)）去趋势 + 循环块置换的 max Δ 分布。"""
    n = len(y)
    beta = 2.0 * np.log(n) * float(np.var(y))
    segs = L.pelt(y, beta=beta, minseglen=minseglen)
    bounds = np.concatenate(([0], segs + 1, [n]))
    r = y.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            r[a:b] -= y[a:b].mean()
    Lb = int(max(8, min(n // 10, round(2 * n ** (1 / 3)))))
    S = np.empty(B)
    for b in range(B):
        S[b] = L.max_delta(L.block_permute(r, Lb, rng))
    return S, len(segs)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    fig = {}
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except (FileNotFoundError, OSError):
            continue
        if not np.all(np.isfinite(y)) or n < 10:  # 与 run_tcpd_main 同口径（41 条；centralia n=15 保留）
            continue
        d = delta_profile(y)
        med = float(np.median(d))
        q90 = float(np.quantile(d, 0.90))
        q99 = float(np.quantile(d, 0.99))
        thr = 3.0 * np.log(n)                      # MBIC per-split 阈值（z-scored 下的工作点）
        Sp = parametric_null(n, float(np.var(y)), B, rng)
        Sq, k_pilot = permutation_null(y, B, rng)
        rows.append({
            "dataset": ds, "n": n,
            "median_delta": med, "q90_delta": q90, "q99_delta": q99,
            "max_delta": float(np.max(d)),
            "mbic_threshold": float(thr),
            "param_null_max_mean": float(np.mean(Sp)),
            "param_null_max_median": float(np.median(Sp)),
            "perm_null_max_mean": float(np.mean(Sq)),
            "perm_null_max_median": float(np.median(Sq)),
            "pilot_k": int(k_pilot),
            "ratio_vs_param": med / float(np.mean(Sp)),
            "ratio_vs_perm": med / float(np.mean(Sq)),
            "ratio_vs_threshold": med / thr,
        })
        fig[ds] = {"profile_sorted": np.sort(d).tolist(),
                   "param_null_maxes": Sp.tolist(),
                   "perm_null_maxes": Sq.tolist(),
                   "mbic_threshold": float(thr)}
        print(f"{ds:22s} n={n:5d} medΔ={med:10.3f} 3ln n={thr:6.2f} "
              f"paramNullMax={np.mean(Sp):7.3f} permNullMax={np.mean(Sq):7.3f} "
              f"ratio(param)={med/np.mean(Sp):8.2f} ratio(perm)={med/np.mean(Sq):8.2f}", flush=True)

    OUT.write_text(json.dumps(rows))
    OUT_FIG5.write_text(json.dumps(fig))

    print(f"\n=== 汇总（n={len(rows)} 数据集）===")
    for key in ["ratio_vs_param", "ratio_vs_perm", "ratio_vs_threshold"]:
        v = np.array([r[key] for r in rows])
        print(f"{key:22s}: min={v.min():8.2f} q25={np.quantile(v,.25):8.2f} "
              f"median={np.median(v):8.2f} q75={np.quantile(v,.75):8.2f} max={v.max():9.2f} "
              f"| <1 的数据集数={int((v<1).sum())}")
    print(f"{'param_null_max_mean':22s}: median={np.median([r['param_null_max_mean'] for r in rows]):7.3f} "
          f"（理论 O(ln ln n) 尺度，n≈10^3 时应为个位数到十几）")
    print(f"{'perm_null_max_mean':22s}: median={np.median([r['perm_null_max_mean'] for r in rows]):7.3f}")
    print(f"{'pilot_k 中位数':22s}: {np.median([r['pilot_k'] for r in rows]):.1f}（若≈0 说明 pilot 未退化）")
    print("\nsaved", OUT, OUT_FIG5)


if __name__ == "__main__":
    main()
