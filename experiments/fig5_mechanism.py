# -*- coding: utf-8 -*-
"""fig5_mechanism.py — 机理图（v2）：真实序列 Δ 背景 vs 两个非退化 null vs MBIC 阈值。

数据源 `results/raw/fig5_data.json` + `results/raw/mismatch.json`
（均由 experiments/diagnose_mismatch.py 生成，EXP-MISMATCH-001）。

v1 的缺陷（review_round3 B5）：v1 的块置换 null 用 β_pilot = 2·ln n·σ̂²（σ̂ = 一阶差分 MAD），
而 σ̂ 在平滑趋势序列上结构性塌陷为 0（差分近似恒定 → MAD≈0）→ pilot 过分割 → 残差恒 0
→ null max 恰为 0（实测 us_population 的 B=100 个值全为 0.0000），比值无意义。

v2 画三条参照线，避免把结论压在一个易碎的比值上：
  - 参数化 null（iid N(0, var(y))）—— MBIC 罚参实际假设的模型，最直接
  - 块置换 null（pilot 罚参改为总方差锚定 β=2·ln n·var(y)，绝不退化）
  - MBIC 的 per-split 工作阈值 3·ln n
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_FIG = ROOT / "results/figures/fig5_mechanism.png"
SRC = ROOT / "results/raw/fig5_data.json"
MISMATCH = ROOT / "results/raw/mismatch.json"

# 三个代表：趋势型（背景最高）/ 金融型 / 背景低于 null 的反例
SHOW = ["us_population", "bitcoin", "quality_control_5"]


def main():
    data = json.loads(SRC.read_text()) if SRC.exists() else {}
    meta = {r["dataset"]: r for r in json.loads(MISMATCH.read_text())} if MISMATCH.exists() else {}
    fig, axes = plt.subplots(1, len(SHOW), figsize=(11, 3.4))
    for ax, ds in zip(np.atleast_1d(axes), SHOW):
        if ds not in data:
            ax.set_visible(False)
            continue
        d = np.array(data[ds]["profile_sorted"])
        Sp = np.array(data[ds]["param_null_maxes"])
        Sq = np.array(data[ds]["perm_null_maxes"])
        thr = float(data[ds]["mbic_threshold"])
        x = np.arange(1, len(d) + 1)
        ax.plot(x, np.maximum(d, 1e-6), lw=1.2, color="#1f4e79",
                label=r"real $\Delta(s)$ profile (sorted)")
        ax.axhline(thr, color="#2e7d32", ls="-", lw=1.4, label=r"MBIC threshold $3\ln n$")
        ax.axhspan(np.min(Sp), np.max(Sp), color="#c0504d", alpha=0.25,
                   label="parametric null, max $\\Delta$ (B=200)")
        ax.axhspan(np.min(Sq), np.max(Sq), color="#4472c4", alpha=0.25,
                   label="block-permutation null, max $\\Delta$ (B=200)")
        ax.set_yscale("log")
        ax.set_xlabel("split candidates (sorted by $\\Delta$)")
        m = meta.get(ds, {})
        sub = (f"median $\\Delta$={m.get('median_delta', float(np.median(d))):.1f} | "
               f"$3\\ln n$={thr:.1f} | "
               f"vs param null {m.get('ratio_vs_param', float('nan')):.1f}$\\times$")
        ax.set_title(f"{ds} (n={m.get('n', len(d))})\n{sub}", fontsize=8)
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("Real split-gain background vs noise-level nulls and the MBIC operating "
                 "threshold (log scale; z-scored series)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    print("saved", OUT_FIG)


if __name__ == "__main__":
    main()
