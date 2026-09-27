# -*- coding: utf-8 -*-
"""
diagnose_sigmahat.py — σ̂（mad_sigma）偏差的闭式诊断。

背景：论文 §5.3 与 §7 有两处明确写着「σ̂ 在强正相关下低估有效检测尺度，
**此解释未经独立验证**」。本脚本把该假设升级为闭式结论。

推导：
  设 y 为 AR(1)（水平序列自相关 φ），边际 sd = σ。则
      var(Δy) = 2σ²(1 − φ)
  mad_sigma 取 MAD(Δy)（用 1.4826 校准到 sd）后除以 √2，故其期望为
      σ̂ = sd(Δy)/√2 = σ·√(1 − φ)
  即 **σ̂/σ = √(1 − φ)**，与 n、σ、amp 无关。

  mad_sigma 里那段"AR 校正"用 ρ̂ = 差分序列的滞后-1 自相关，而 AR(1) 下
      ρ_d = −(1 − φ)/2 ≤ 0
  经 `max(0, ·)` 后恒为 0 → **校正结构性不触发**（不是调得不好，是用错了量）。
  这就是 σ̂ 偏低且无法自愈的原因。

验证：在 φ ∈ {0.5, 0.9} 上同时测 (i) σ̂/σ 与 √(1−φ) 的吻合度；(ii) ρ_d 与 −(1−φ)/2
的吻合度。两项都吻合则闭式成立。

输出：results/raw/sigmahat_closedform.json（覆盖写；纯确定性计算，几秒完成）
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import synthetic as S  # noqa: E402

OUT = ROOT / "results" / "raw" / "sigmahat_closedform.json"

CASES = [
    # (noise 名, φ)
    ("ar1_05", 0.5),
    ("ar1_09", 0.9),
]
# 长序列、无变点（k=0）→ 纯噪声，ρ_d 估计干净
RHO_N = 20000
RATIO_N = 5000
RATIO_K = 5
RATIO_AMP = 3.0
RATIO_SIGMAS = [1.0, 10.0]
RATIO_REPS = 3


def main():
    out = {"cases": [], "note": "σ̂/σ 预测 = sqrt(1-φ)；ρ_d 预测 = -(1-φ)/2"}

    for nz, phi in CASES:
        # (i) σ̂/σ
        ratios = []
        for s in RATIO_SIGMAS:
            for rep in range(RATIO_REPS):
                y, _ = S.gen_series(RATIO_N, RATIO_K, nz, RATIO_AMP, s, seed=1000 + rep)
                ratios.append(float(L.mad_sigma(y) / s))
        # (ii) ρ_d
        y0, _ = S.gen_series(RHO_N, 0, nz, 0.0, 1.0, seed=7)
        d = np.diff(y0)
        rho_d = float(np.corrcoef(d[:-1], d[1:])[0, 1])

        out["cases"].append({
            "noise": nz, "phi": phi,
            "ratio_pred": float(np.sqrt(1 - phi)),
            "ratio_meas_mean": float(np.mean(ratios)),
            "ratio_meas_sd": float(np.std(ratios)),
            "ratio_n": len(ratios),
            "rho_d_pred": float(-(1 - phi) / 2),
            "rho_d_meas": rho_d,
            "rho_d_after_max0": float(max(0.0, rho_d)),
        })

    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'noise':8s} {'φ':>5s} {'σ̂/σ 预测':>10s} {'σ̂/σ 实测':>16s} "
          f"{'ρ_d 预测':>10s} {'ρ_d 实测':>10s} {'max(0,ρ_d)':>11s}")
    for c in out["cases"]:
        print(f"{c['noise']:8s} {c['phi']:5.1f} {c['ratio_pred']:10.4f} "
              f"{c['ratio_meas_mean']:9.4f}±{c['ratio_meas_sd']:.4f} "
              f"{c['rho_d_pred']:10.4f} {c['rho_d_meas']:10.4f} {c['rho_d_after_max0']:11.4f}")
    print("\n结论：若两列实测都贴近预测，则 σ̂ 的 AR 偏差是**结构性**的"
          "（σ̂=σ√(1−φ)，且 max(0,·) 校正恒不触发），而非调参问题。")


if __name__ == "__main__":
    main()
