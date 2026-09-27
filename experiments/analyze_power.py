# -*- coding: utf-8 -*-
"""
analyze_power.py — §7 功效数字的固化脚本（回应外部评审 Q-01）。

论文 §7 引用的三个数字此前为手工计算：
  n=41、sd=0.0855 时，ΔF1=0.0122 的功效 0.14；ΔF1=0.02 的功效 0.31；
  ΔF1=0.02 要达到 0.91 需要 n=200。
本脚本把计算口径固化为可执行、可复验的形式。

口径（与论文数字完全一致，双侧 α=0.05）：
- 配对 t 检验功效，非中心 t 分布精确计算（与 R power.t.test 同口径）；
- sd=0.0855 取自 matched-k aggregate（k=2..5 数据集内平均）差值的观测 sd。
  **sd 为样本 sd（ddof=1）**：总体 sd（ddof=0）为 0.0845，对应功效
  0.147256/0.315699/0.914886 → 取两位 **0.15/0.32/0.91**，而论文印的是
  0.14/0.31/0.91 ⇒ **三格中前两格依赖 ddof=1 口径**（第三格 n=200 落在 0.915
  舍入边界上，两种口径都入到 0.91，不具判别力）。该对照在 main() 里以断言固化，
  不是散文：口径若被悄悄改掉，脚本立即失败。
- 两个输入（ΔF1=0.0122、sd=0.0855）**不是手抄常数**：main() 启动时从
  results/raw/matched_k.jsonl 重算 k=2..5 聚合并逐项断言（见 analyze_matched_k.py 表 D）。
  抄错任何一个字面量都会在此处立即失败。
限定（如实声明）：这是 Wilcoxon 语境下的参数近似（渐近相对效率≈0.955，
结论方向不变），不是 Wilcoxon 功效本身的精确值。

输出：控制台 + results/raw/power.json（记录输入、口径与输出）。
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "raw" / "power.json"

ALPHA = 0.05
SD = 0.0855
DELTA_QUOTED = 0.0122
N_QUOTED = 41
KS_AGG = (2, 3, 4, 5)

# ddof 对照的期望值：用总体 sd（ddof=0）重算三格功效，是否偏离论文印出的值。
#   True  -> 该格确实依赖 ddof=1 口径
#   None  -> 该格落在舍入边界上，不具判别力，**不作断言**（见 docstring）
DDOF0_DIFFERS = (True, True, None)


def derive_matched_k_aggregate():
    """从 results/raw/matched_k.jsonl 重算 (n, ΔF1, sd_ddof1)。

    与论文正文「averaging over k = 2..5 within each dataset」同口径：
    先在数据集内对 k∈KS_AGG 取 greedy / optimal 的 F1 平均，再作差，再跨数据集聚合。
    """
    rows = [json.loads(l) for l in
            (ROOT / "results" / "raw" / "matched_k.jsonl").read_text().splitlines()
            if l.strip()]
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["minseglen"] == 2:
            per[r["dataset"]][r["k"]][r["rule"]] = r
    g, o = [], []
    for d in sorted(per):
        gs = [per[d][k]["greedy"]["f1"] for k in KS_AGG
              if k in per[d] and "greedy" in per[d][k]]
        os_ = [per[d][k]["optimal"]["f1"] for k in KS_AGG
               if k in per[d] and "optimal" in per[d][k]]
        if len(gs) != len(KS_AGG) or len(os_) != len(KS_AGG):
            continue
        g.append(float(np.mean(gs)))
        o.append(float(np.mean(os_)))
    diff = np.array(g) - np.array(o)
    return len(diff), float(diff.mean()), float(diff.std(ddof=1))


def paired_t_power(n, delta, sd=SD, alpha=ALPHA):
    """Two-sided paired t-test power via the non-central t distribution."""
    ncp = delta * np.sqrt(n) / sd
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return float((1 - stats.nct.cdf(crit, n - 1, ncp))
                 + stats.nct.cdf(-crit, n - 1, ncp))


def main():
    # --- 输入绑定：从 raw 重算，断言论文引用的两个字面量 ---
    n_agg, delta_agg, sd_agg = derive_matched_k_aggregate()
    print(f"matched-k aggregate (k=2..5, ddof=1): "
          f"n={n_agg} delta={delta_agg:.6f} sd={sd_agg:.6f}")
    assert n_agg == N_QUOTED, f"n {n_agg} != {N_QUOTED}"
    assert round(delta_agg, 4) == DELTA_QUOTED, f"delta {delta_agg:.6f} != {DELTA_QUOTED}"
    assert round(sd_agg, 4) == SD, f"sd {sd_agg:.6f} != {SD}"

    # Δ=0.02 / n=200 是反事实场景参数（stipulated scenario），不是测量值，
    # 原则上无法从 raw 派生，字面量在此是必须的（见 skill §40）。
    cases = [
        {"n": 41, "delta": 0.0122, "quoted": 0.14},
        {"n": 41, "delta": 0.02, "quoted": 0.31},
        {"n": 200, "delta": 0.02, "quoted": 0.91},
    ]
    out = {"alpha": ALPHA, "sd": SD,
           "method": "two-sided paired t power, exact non-central t "
                     "(same convention as R power.t.test); parametric "
                     "approximation to the Wilcoxon setting (ARE~0.955)",
           "cases": []}
    ok = True
    for c in cases:
        p = paired_t_power(c["n"], c["delta"])
        match = round(p, 2) == c["quoted"]
        ok = ok and match
        out["cases"].append({**c, "computed": round(p, 4),
                             "matches_quoted": match})
        print(f"n={c['n']:>3d} delta={c['delta']:.4f}: power={p:.4f} "
              f"(quoted {c['quoted']:.2f}) {'OK' if match else 'MISMATCH'}")
    # --- ddof 对照：把「口径敏感性」从散文变成断言 ---
    # 用总体 sd（ddof=0）重算；期望见 DDOF0_DIFFERS。口径一旦被改掉，
    # 或第三格脱离边界，这里立即失败。
    sd0 = float(sd_agg * np.sqrt((n_agg - 1) / n_agg))
    print(f"[ddof=0 counterfactual] sd={sd0:.6f} (vs sample sd {sd_agg:.6f})")
    for c, expect in zip(cases, DDOF0_DIFFERS):
        p0 = paired_t_power(c["n"], c["delta"], sd0)
        print(f"  n={c['n']:>3d} delta={c['delta']:.4f}: power={p0:.4f} "
              f"(quoted {c['quoted']:.2f})")
        if expect is not None:
            assert (round(p0, 2) != c["quoted"]) is expect, (
                f"ddof=0 counterfactual no longer holds for "
                f"n={c['n']} delta={c['delta']}: {p0:.6f}")

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("saved", OUT)
    if not ok:
        raise SystemExit("power numbers do not reproduce - investigate")


if __name__ == "__main__":
    main()
