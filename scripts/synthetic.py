# -*- coding: utf-8 -*-
"""
synthetic.py — 受控合成数据生成器（分段常数均值 + 可控噪声）。

因子：n, k(变点数), noise, amp(幅值/σ), sigma(噪声尺度), seed。

noise 取值：
  - ``gauss``   N(0, σ²)
  - ``t3``      σ·t(3)/√3（边际方差 σ²）
  - ``ar1_<dd>`` 平稳 AR(1)，φ = dd/10，边际方差 σ²。
    ``dd ∈ {00, 02, 05, 07, 09}`` ⇒ φ ∈ {0, 0.2, 0.5, 0.7, 0.9}。
    旧名 ``ar1_09`` / ``ar1_05`` 保持不变（数值与以前逐位一致）。
所有随机性由 seed 控制，可复现。变点位置 0-based（左段末点下标），与全库一致。
"""

from __future__ import annotations

import numpy as np


def gen_series(
    n: int,
    k: int,
    noise: str = "gauss",
    amp: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0,
):
    """生成分段常数均值序列。

    - 变点位置：均匀随机，最小段长 m0 = max(10, n // (3*(k+1)))
    - 段均值：随机游走，每步 ±U(0.5,1)*amp*sigma（保证相邻段有差异）
    - noise: gauss = N(0,σ²)；t3 = σ·t(3)/√3；ar1_09 = φ=0.9 平稳 AR(1)，边际方差 σ²
    返回 (y, cps_0based)
    """
    rng = np.random.default_rng(seed)
    if k > 0:
        # 分层放置：k+1 个等分槽位，每个变点在槽内 ±25% 抖动
        # （相邻最小间距 = 0.5·w ≥ n/(3(k+1)) ≥ m0，恒可行）
        w = n / (k + 1)
        pos = np.round((np.arange(k) + 0.5) * w
                       + rng.uniform(-0.25, 0.25, size=k) * w).astype(int)
        pos = np.clip(pos, 1, n - 2)
        pos = np.unique(pos)
    else:
        pos = np.array([], dtype=int)

    # 段均值（随机游走）
    nseg = k + 1
    steps = rng.uniform(0.5, 1.0, size=nseg) * amp * sigma * rng.choice([-1, 1], size=nseg)
    means = np.cumsum(steps)
    bounds = np.concatenate(([0], pos + 1, [n]))
    y = np.empty(n)
    for i in range(nseg):
        y[bounds[i] : bounds[i + 1]] = means[i]

    # 噪声
    if noise == "gauss":
        eps = rng.normal(0.0, sigma, size=n)
    elif noise == "t3":
        eps = sigma * rng.standard_t(3, size=n) / np.sqrt(3.0)
    elif noise.startswith("ar1_"):
        # ar1_<dd> —— φ = dd/10。φ=0 时退化为 N(0,σ²)（与 gauss 分布相同，
        # 但走 AR(1) 分支以便 φ 扫描在同一条代码路径上，排除实现差异这个混淆项）。
        phi = int(noise[len("ar1_"):]) / 10.0
        if not (0.0 <= phi < 1.0):
            raise ValueError("ar1 phi out of range: %s" % noise)
        sd_innov = sigma * np.sqrt(1 - phi**2)
        e = rng.normal(0.0, sd_innov, size=n)
        if phi == 0.0:
            eps = e
        else:
            eps = np.empty(n)
            eps[0] = e[0] / np.sqrt(1 - phi**2)
            for t in range(1, n):
                eps[t] = phi * eps[t - 1] + e[t]
    else:
        raise ValueError(noise)
    return y + eps, pos.tolist()
