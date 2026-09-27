# -*- coding: utf-8 -*-
"""
cpd_lib.py — 变点检测核心库（zcode2026）

两个层次：
A) TCPDBench/R changepoint 的忠实复现（验证基准，依据 C/R 源码逐行核对，2026-09-13）
B) 本文方法 SC-PELT 及其构件

R changepoint 2.2.4 语义备忘（cpt.mean, test.stat="Normal", z-scored 数据）：
- mbic_mean 代价 = SSE(段) + log(n)（C: cost_general_functions.c:58）
- PELT: F(t) = min_s F(s) + mbic_cost(s..t) + pen, F(0) = -pen, pen = 3*log(n)
  → 每 CP 有效阈值 Δ_SSE ≥ 4*log(n)
- BinSeg(.C('binseg')): 无阈值递归分裂 Q 轮（每轮取全局 argmax lambda 的分裂），
  lambda(c) = 0.5*Δ(c) − 0.5*(ln m1 + ln m2 − ln m)，
  likeout[q] = min(likeout[q-1], max lambda_q)（瓶颈），
  接受最长前缀 s.t. 2*likeout[q] >= pen ⇔ 前缀内每个分裂
  Δ >= pen + ln m1 + ln m2 − ln m  （= 3 ln n + ln m1 + ln m2 − ln m）。
  R55 实测订正：旧备忘写「Δ_q >= 4*log(n)」是错的 —— 用 R changepoint 2.2.2
  直接判定一条判别序列（n=131, Δ_first=18.4154）：R 接受该分裂，与
  「3lnn+lnm1+lnm2−lnm = 17.9357」一致，与「4 ln n = 19.5008」相反。
  候选 1-based j ∈ [st+1, end-1] ∩ [2, n-3]（左段 ≥2 点、右段 ≥1 点的 C 侧不对称约束）
- AMOC: 先用裸 Δ 选最优分裂，再判 Δ − ln(m1) − ln(m2) ≥ pen（pen = 3*log(n)），
  即 Δ ≥ 3 ln n + ln m1 + ln(n−cpt+1)；m2 = n − c（R 侧 log(n−cpt+1) 的 quirk）。
  R55 实测订正：旧备忘写「Δ_SSE ≥ 3*log(n)」漏掉了两个 ln 项 —— R 2.2.2 在
  判别序列（n=108, Δ=15.4574, pen=14.0464）上返回 ncpts=0，与含 ln 项的公式一致。
- SegNeigh: k ∈ 0..Q-1 精确 DP，criterion = SSE(k) + k*BIC, BIC = 2*log(n)
- R 导出时 cpts 尾部附加哨兵 n → 导出的 cplocations 末尾恒有 n-1（幻影变点），
  仅 PELT/BinSeg/SegNeigh（AMOC 不加）。

变点位置约定：0-based，左段末点下标。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 基础工具


def zscore(y: np.ndarray) -> np.ndarray:
    """与 R utils.R scale(mat) 一致（样本标准差, ddof=1）。"""
    y = np.asarray(y, dtype=float)
    mu = y.mean()
    sd = y.std(ddof=1)
    if sd == 0:
        return np.full_like(y, np.nan)
    return (y - mu) / sd


class SumStats:
    """前缀和工具：O(1) 查询区间 SSE。点下标 0..n-1，切片语义 [a, b) 左闭右开。"""

    def __init__(self, y: np.ndarray):
        self.y = y
        self.n = len(y)
        self.s1 = np.concatenate(([0.0], np.cumsum(y)))
        self.s2 = np.concatenate(([0.0], np.cumsum(y * y)))

    def sse(self, a: int, b: int) -> float:
        """SSE of y[a:b]（b > a）。"""
        s = self.s1[b] - self.s1[a]
        q = self.s2[b] - self.s2[a]
        m = b - a
        return q - s * s / m

    def sse_vec(self, a, b):
        """向量化 SSE：区间 [a,b)，a/b 可为标量或同形数组。"""
        s = self.s1[b] - self.s1[a]
        q = self.s2[b] - self.s2[a]
        m = np.asarray(b) - np.asarray(a)
        return q - s * s / m


# ---------------------------------------------------------------- PELT（标准、精确）


def pelt(y: np.ndarray, beta: float, minseglen: int = 1) -> np.ndarray:
    """标准 PELT，SSE 代价 + 每 CP 罚参 beta。对任意 beta>=0 精确。

    返回 0-based 变点（左段末点下标，升序）。
    """
    ss = SumStats(y)
    n = ss.n
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    admissible = np.zeros(n + 1, dtype=bool)
    cp = np.full(n + 1, -1, dtype=np.int64)

    for t in range(1, n + 1):
        admissible[t - 1] = True
        cand = np.nonzero(admissible[:t])[0]
        cand = cand[t - cand >= minseglen]
        if cand.size == 0:
            F[t] = np.inf
            continue
        s1b = ss.s1[t]
        s2b = ss.s2[t]
        sse_cand = (s2b - ss.s2[cand]) - (s1b - ss.s1[cand]) ** 2 / (t - cand)
        costs = F[cand] + sse_cand + beta
        i = int(np.argmin(costs))
        F[t] = costs[i]
        cp[t] = cand[i]
        admissible[cand] = (F[cand] + sse_cand) < F[t]

    t = n
    cps = []
    while t > 0:
        s = cp[t]
        if s <= 0:
            break
        cps.append(s - 1)
        t = s
    return np.array(sorted(cps), dtype=int)


# ---------------------------------------------------------------- BinSeg（R changepoint C 语义，忠实复现）


def binseg_r(y: np.ndarray, pen: float, Q: int = 5) -> np.ndarray:
    """TCPDBench default_binseg 的忠实复现（不含幻影 n-1）。

    pen = 3*log(n)（MBIC）。C 语义（BinSeg_one_func_minseglen.c）：
    - 无阈值贪心递归分裂 Q 轮，每轮对当前全部分段取 argmax lambda；
      lambda(c) = 0.5*Δ(c) − 0.5*(ln m1 + ln m2 − ln m)（mbic 代价内嵌 log 段长）
      其中 m=段长, m1/m2=分裂后两段长；候选 1-based j ∈ [st+1, end-1] ∩ [2, n-3]
      （左段 ≥2 点、右段 ≥1 点，C 侧不对称；0-based: c ∈ [a+1, b-2] ∩ [1, n-4]）
    - likeout[q] = min(likeout[q-1], maxout_q)（瓶颈），oldmax 初始 +inf，
      lambda[0]=lambda[1]=0（候选区外恒 0）
    - 接受最长前缀 s.t. 2*likeout[q] >= pen ⇔ 前缀内每个分裂 Δ >= pen + ln m1 + ln m2 − ln m
    """
    ss = SumStats(y)
    n = ss.n
    if n < 4:
        return np.array([], dtype=int)
    logn = np.log(n)
    segs = [(0, n)]  # [a, b) 0-based
    greedy: list[tuple[int, float]] = []  # (CP_0based, likeout_value)
    oldmax = np.inf
    for _q in range(Q):
        lam = np.zeros(n)  # lambda[0..n-1]，候选区外为 0
        best = None  # (lambda, c)
        for (a, b) in segs:
            lo = max(a + 1, 1)
            hi = min(b - 2, n - 4)
            if hi < lo:
                continue
            c_arr = np.arange(lo, hi + 1)  # 0-based 分裂点 c
            m = b - a
            m1 = c_arr + 1 - a
            m2 = b - (c_arr + 1)
            sse_full = ss.sse(a, b)
            d = sse_full - ss.sse_vec(a, c_arr + 1) - ss.sse_vec(c_arr + 1, b)
            lam_c = 0.5 * d - 0.5 * (np.log(m1) + np.log(m2) - np.log(m))
            lam[c_arr] = lam_c
            j = int(np.argmax(lam_c))  # first max
            if best is None or lam_c[j] > best[0]:
                best = (float(lam_c[j]), int(c_arr[j]))
        if best is None:
            # 全部段无候选 → lambda 全 0 → argmax=0（C 行为：whichout=0）
            like_q = min(oldmax, 0.0)
            oldmax = like_q
            greedy.append((0, like_q))
            continue
        maxout, c = best
        like_q = min(oldmax, maxout)
        oldmax = like_q
        greedy.append((c, like_q))
        for i, (a, b) in enumerate(segs):
            if a <= c < b:
                segs[i] = (a, c + 1)
                segs.insert(i + 1, (c + 1, b))
                break
    out = []
    for c, like in greedy:
        if 2.0 * like >= pen:
            out.append(c)
        else:
            break
    return np.array(sorted(out), dtype=int)


# ---------------------------------------------------------------- AMOC（R 语义）


def amoc(y: np.ndarray, pen: float) -> np.ndarray:
    """AMOC/MBIC 忠实复现：先用裸 Δ 选最优分裂（first max），再判
    Δ − ln(m1) − ln(m2) >= pen（pen = 3*log(n)）。"""
    ss = SumStats(y)
    n = ss.n
    if n < 2:
        return np.array([], dtype=int)
    c_arr = np.arange(0, n - 1)  # 0-based CP ∈ [0, n-2]
    d = ss.sse(0, n) - ss.sse_vec(0, c_arr + 1) - ss.sse_vec(c_arr + 1, n)
    j = int(np.argmax(d))
    m1 = c_arr[j] + 1          # R tmp[1] = cpt（1-based 左段长）
    m2 = n - c_arr[j]          # R: log(n - cpt + 1) → n - cpt + 1 = n - c（R 原样的 quirk）
    stat = d[j] - np.log(m1) - np.log(m2)
    if stat >= pen:
        return np.array([c_arr[j]], dtype=int)
    return np.array([], dtype=int)


# ---------------------------------------------------------------- PELT（R changepoint MBIC 语义，忠实复现）


def pelt_mbic(y: np.ndarray, pen: float) -> np.ndarray:
    """TCPDBench default_pelt 的忠实复现（C: PELT_one_func_minseglen.c）。

    代价 cost(s,t) = SSE(s..t-1) + ln(t-s)（mbic_mean，log 为段长）；
    F[0] = -pen；F[t] = min F[s] + cost + pen；
    剪枝（照抄 C）：保留 s iff F[s] + cost(s,t) <= F[t] + pen；
    候选初值 {0, 1}，每步加入边界 t-1（C: tstar-(minseglen-1), minseglen=1）。
    该剪枝对含 log 段长的代价类不保证精确——如实复现 R 行为（含其次优解）。
    """
    ss = SumStats(y)
    n = ss.n
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    cp = np.full(n + 1, -1, dtype=np.int64)
    checklist = [0, 1] if n >= 2 else [0]
    if n >= 2:
        F[1] = ss.sse(0, 1) + np.log(1)  # C: lastchangelike[j] = costfunction(...)

    for t in range(2, n + 1):
        cand = np.array([s for s in checklist if t - s >= 1], dtype=int)
        if cand.size == 0:
            continue
        sse_c = ss.sse_vec(cand, t)
        cost_c = sse_c + np.log(t - cand)
        tmplike = F[cand] + cost_c
        i = int(np.argmin(tmplike))
        F[t] = tmplike[i] + pen
        cp[t] = cand[i]
        # 照抄 C 剪枝：保留 F[s] + cost <= F[t] + pen
        keep = tmplike <= (F[t])
        checklist = [int(s) for s, k in zip(cand, keep) if k]
        checklist.append(t - 1)

    t = n
    cps = []
    while t > 0:
        s = cp[t]
        if s <= 0:
            break
        cps.append(s - 1)
        t = s
    return np.array(sorted(cps), dtype=int)


# ---------------------------------------------------------------- SegNeigh（R 语义，BIC）


def segneigh(y: np.ndarray, beta: float, Q: int = 5) -> np.ndarray:
    """k ∈ 0..Q-1 精确 DP，criterion = SSE(k) + k*beta，取最小（平局取第一）。

    R default_segneigh: beta = 2*log(n)（BIC）。返回 0-based 变点（不含幻影）。
    """
    ss = SumStats(y)
    n = ss.n
    if n < 4:
        return np.array([], dtype=int)
    Best = np.full((Q, n + 1), np.inf)
    Best[0, 0] = 0.0
    cp = np.full((Q, n + 1), -1, dtype=np.int64)
    for j in range(1, n + 1):
        Best[0, j] = ss.sse(0, j)
    for qi in range(1, Q):  # qi = 变点个数
        for j in range(qi + 1, n + 1):
            v = np.arange(max(qi - 1, 0), j)  # 最后一段 [v, j)，左 v 点 qi-1 个 CP
            vals = Best[qi - 1, v] + ss.sse_vec(v, j)
            i = int(np.argmin(vals))
            Best[qi, j] = vals[i]
            cp[qi, j] = v[i]
    k = np.arange(Q)
    criterion = Best[:, n] + k * beta
    kbest = int(np.argmin(criterion))
    cps, j, q = [], n, kbest
    while q > 0:
        v = cp[q, j]
        cps.append(v - 1)
        j = v
        q -= 1
    return np.array(sorted(cps), dtype=int)


# ---------------------------------------------------------------- Zero


def zero(y: np.ndarray) -> np.ndarray:
    return np.array([], dtype=int)


# ---------------------------------------------------------------- 统计量工具


def max_delta(y: np.ndarray, ss: SumStats | None = None) -> float:
    """全序列最优单分裂的 SSE 下降量 max_s Δ(s)。O(n)。"""
    if ss is None:
        ss = SumStats(y)
    n = ss.n
    if n < 2:
        return 0.0
    c_arr = np.arange(0, n - 1)
    d = ss.sse(0, n) - ss.sse_vec(0, c_arr + 1) - ss.sse_vec(c_arr + 1, n)
    return float(np.max(d))


def mad_sigma(y: np.ndarray) -> float:
    """鲁棒噪声尺度：一阶差分 MAD + 滞后1自相关校正（AR(1) 修正）。

    var(diff) ≈ 2σ²(1−ρ)；先用 MAD 得稳健 var(diff)，再用差分的滞后-1
    相关系数 ρ̂≥0 校正：σ̂² = var_diff_robust / (2(1−ρ̂))。
    """
    d = np.diff(y)
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med))
    var_diff = (mad / np.sqrt(2.0)) ** 2  # = σ²(1−ρ)（AR(1) 下）
    if len(d) > 2:
        c = np.corrcoef(d[:-1], d[1:])[0, 1]
        rho = max(0.0, 0.0 if not np.isfinite(c) else c)
        if 0.0 <= rho < 1.0:
            var_diff = var_diff / (1.0 - rho)  # σ² = var(diff)/(2(1−ρ))，其中 var_diff 已含 /2
    return float(np.sqrt(max(var_diff, 1e-12)))


def block_permute(r: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    """循环块置换：随机起点、块长 L，拼接后截断到 n。O(n)。"""
    n = len(r)
    nblocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nblocks)
    out = np.empty(nblocks * L, dtype=float)
    for i, s in enumerate(starts):
        idx = (s + np.arange(L)) % n
        out[i * L : (i + 1) * L] = r[idx]
    return out[:n]


# ---------------------------------------------------------------- SC-PELT（本文方法，内部代号，命名最后定）


def sc_pelt(
    y: np.ndarray,
    B: int = 100,
    alpha: float = 0.01,
    L: int | None = None,
    beta_pilot: float = 0.5,
    max_iter: int = 5,
    seed: int = 12345,
    use_floor: bool = True,
    pilot_mult: float = 2.0,
    return_info: bool = False,
):
    """置换校准罚参 + 自洽 Bonferroni 的 PELT。

    use_floor=True（默认）：β = max(β_perm, 2·ln(n)·var(y)) 的保守组合地板。
    z-score 序列 var=1 → 与 TCPD 口径一致；原始尺度序列地板自动随总方差缩放。
    use_floor=False：纯置换校准（消融用；仅适用于近白噪声、模型正确的场景）。
    pilot_mult：pilot 过分割罚参 = pilot_mult·ln(n)·σ̂²（BIC 尺度；v2 修复——
    过小的 pilot 会把噪声吃进段均值、压低 null 尺度，见 FAILURES.md #F2）。
    返回 0-based 变点（return_info 时附诊断 dict）。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    ss = SumStats(y)
    sigma = mad_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    if L is None:
        L = int(max(8, min(n // 10, round(2 * n ** (1 / 3)))))
    segs = pelt(y, beta=pilot_mult * np.log(n) * sigma**2, minseglen=2)  # BIC 尺度 pilot
    bounds = np.concatenate(([0], segs + 1, [n]))
    r = y.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            r[a:b] -= y[a:b].mean()

    S = np.empty(B)
    for b in range(B):
        yb = block_permute(r, L, rng)
        S[b] = max_delta(yb)

    k_prev = None
    # v3：置换校准 β_perm 与「总方差锚定 BIC 地板」β_floor 的保守组合。
    # 机理（见 literature/04_mechanism_notes.md）：
    #  - 残差置换 null 只刻画白噪声尺度，无法覆盖趋势/季节性导致的 Δ 背景抬升；
    #  - z-score 后总方差≈1，BIC 地板 β=2·ln(n)·σ̂²_total 是失配容忍下界；
    #  - max 组合：噪声正确时置换项主导（近最优），失配时地板兜底。
    var_total = float(np.var(y))
    beta_floor = 2.0 * np.log(n) * var_total if use_floor else 0.0
    info = {"beta_floor": beta_floor, "sigma": sigma, "use_floor": use_floor}
    taus = np.array([], dtype=int)
    for it in range(max_iter):
        k = max(1, len(taus))
        q = min(1 - alpha / k, 1.0 - 1e-12)
        beta_perm = float(np.quantile(S, q))
        beta = max(beta_perm, beta_floor)
        taus = pelt(y, beta=beta, minseglen=1)
        info["iters"] = it + 1
        info["beta_last"] = beta
        info["beta_perm_last"] = beta_perm
        info[f"k_iter{it+1}"] = len(taus)
        if k_prev is not None and len(taus) == k_prev:
            break
        k_prev = len(taus)
    if return_info:
        info["L"] = L
        info["beta_first"] = info.get("beta_perm_last")
        return taus, info
    return taus


# ---------------------------------------------------------------- 封装：TCPDBench default 对应物（不含幻影 n-1）


def default_amoc(y):
    return amoc(y, pen=3.0 * np.log(len(y)))


def default_binseg(y):
    return binseg_r(y, pen=3.0 * np.log(len(y)), Q=5)


def default_pelt(y):
    return pelt_mbic(y, pen=3.0 * np.log(len(y)))


def default_segneigh(y):
    return segneigh(y, beta=2.0 * np.log(len(y)), Q=5)
