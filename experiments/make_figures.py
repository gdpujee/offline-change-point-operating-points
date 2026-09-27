# -*- coding: utf-8 -*-
"""make_figures.py — 论文图表（全部由 raw 生成，输出到 results/figures/）。

Fig1: TCPD 各算法 F1/Covering 条形图（误差条=数据集间 std）
Fig2: 合成网格 k 误差热图（algorithm × noise，σ=10 子网格）
Fig3: 敏感性——TCPD F1 vs (B, α, L)，floor on/off
Fig4: 规模——runtime vs n（log-log）
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig1():
    rows = json.loads((ROOT / "results/raw/tcpd_main.json").read_text())
    per_ds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_ds[r["algorithm"]]["f1"].append(np.mean(
            [x["f1"] for x in rows if x["dataset"] == r["dataset"] and x["algorithm"] == r["algorithm"]]))
    algs = ["zero", "segneigh", "pelt_bic", "amoc", "pelt", "binseg", "sc_pelt"]
    means = [np.mean(per_ds[a]["f1"]) for a in algs]
    stds = [np.std(per_ds[a]["f1"]) for a in algs]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(algs)), means, yerr=stds, capsize=3, color="#4472c4")
    ax.set_xticks(range(len(algs)))
    ax.set_xticklabels(algs, rotation=20)
    ax.set_ylabel("F1 (M=5)")
    ax.set_title("TCPD (41 univariate, phantom-free): mean per-dataset F1")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_tcpd_f1.png", dpi=200)
    plt.close(fig)


def fig6():
    """匹配 k 实验：F1 随 k 的变化（两条选择规则）+ 各方法实际 k。"""
    rows = [json.loads(l) for l in (ROOT / "results/raw/matched_k.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["minseglen"] == 2]
    per = defaultdict(lambda: defaultdict(list))  # rule -> k -> F1 list
    for r in rows:
        per[r["rule"]][r["k"]].append(r["f1"])
    ks = sorted({r["k"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    style = {"greedy": ("#1f4e79", "o-", "BinSeg greedy order"),
             "optimal": ("#c0504d", "s--", "SSE optimal (= PELT path)")}
    for rule, (c, m, lbl) in style.items():
        if rule not in per:
            continue
        xs = sorted(per[rule])
        ys = [np.mean(per[rule][k]) for k in xs]
        ax.plot(xs, ys, m, color=c, label=lbl)
    # 默认方法实际平均 k（来自 tcpd_main.json）
    trows = json.loads((ROOT / "results/raw/tcpd_main.json").read_text())
    by = defaultdict(list)
    for r in trows:
        by[r["algorithm"]].append(r["k"])
    method_means = {a: float(np.mean(by[a])) for a in ["zero", "segneigh", "pelt_bic", "sc_pelt", "pelt", "binseg"]}
    method_color = {"zero": "#888888", "segneigh": "#9966cc", "pelt_bic": "#e08000",
                    "sc_pelt": "#39a275", "pelt": "#39a275", "binseg": "#1f4e79"}
    for a, mk in method_means.items():
        ax.axvline(mk, color=method_color.get(a, "#999"), ls=":", alpha=0.7)
    ax.set_xlabel("k (number of changepoints)")
    ax.set_ylabel("mean F1 (M=5)")
    ax.set_title("Matching k on TCPD (41 datasets): F1(k) profile vs each method's mean k")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_matched_k.png", dpi=200)
    plt.close(fig)


def fig2():
    rows = json.loads((ROOT / "results/raw/synth_grid.json").read_text())
    agg = defaultdict(list)
    for r in rows:
        agg[(r["algorithm"], r["noise"], r["sigma"])].append(r["k_pred"] - r["k_true"])
    algs = ["mbic_raw", "binseg_raw", "bic_var", "sc_nofloor", "sc_v3"]
    noises = ["gauss", "t3", "ar1_09"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, sigma in zip(axes, [1.0, 10.0]):
        M_ = np.zeros((len(algs), len(noises)))
        for i, a in enumerate(algs):
            for j, nz in enumerate(noises):
                M_[i, j] = np.mean(agg[(a, nz, sigma)])
        M_ = np.clip(M_, -10, 50)
        im = ax.imshow(M_, cmap="RdBu_r", vmin=-10, vmax=50, aspect="auto")
        ax.set_xticks(range(len(noises)))
        ax.set_xticklabels(noises)
        ax.set_yticks(range(len(algs)))
        ax.set_yticklabels(algs)
        ax.set_title(f"σ={sigma:g}")
        for i in range(len(algs)):
            for j in range(len(noises)):
                ax.text(j, i, f"{M_[i,j]:+.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, label="k error (pred − true), clipped")
    fig.suptitle("Synthetic grid: changepoint-count error")
    fig.savefig(FIG / "fig2_synth_kerr.png", dpi=200)
    plt.close(fig)


def fig3():
    rows = json.loads((ROOT / "results/raw/sensitivity.json").read_text())
    tcpd = [r for r in rows if r["data"].startswith("tcpd")]
    cfgs = sorted({r["config"] for r in tcpd})
    fl_vals = [True, False]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for fl, c in zip(fl_vals, ["#4472c4", "#c0504d"]):
        vals = [np.mean([r["f1"] for r in tcpd if r["config"] == cfg and r["floor"] == fl]) for cfg in cfgs]
        ax.plot(range(len(cfgs)), vals, "o-", color=c, label=f"floor={fl}")
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("TCPD F1")
    ax.set_title("SC-PELT sensitivity: B / alpha / L (TCPD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_sensitivity.png", dpi=200)
    plt.close(fig)


def fig4():
    rows = [json.loads(l) for l in (ROOT / "results/raw/scale.jsonl").read_text().splitlines() if l.strip()]
    agg = defaultdict(list)
    for r in rows:
        agg[r["algorithm"]].append((r["n"], r["runtime_s"]))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for a, v in agg.items():
        v = sorted(v)
        ax.plot([x[0] for x in v], [x[1] for x in v], "o-", label=a)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("runtime (s)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Scalability (gauss, k=5, amp=3σ, σ=1)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_scale.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    try:
        fig4()
    except FileNotFoundError:
        print("scale.json not ready; fig4 skipped")
    try:
        fig6()
    except FileNotFoundError:
        print("matched_k.jsonl not ready; fig6 skipped")
    print("figures saved to", FIG)
