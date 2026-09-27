# -*- coding: utf-8 -*-
"""
metrics_tcpdbench.py — TCPDBench 官方评估指标的忠实移植。

来源：alan-turing-institute/TCPDBench analysis/scripts/metrics.py
（Copyright (c) 2020 - The Alan Turing Institute；为对齐基准口径逐字移植，
仅做中文注释补充。验证：experiments/validate_metrics.py 会与官方文件直接对拍。）

约定：变点位置 0-based（左段末点下标）。
"""

from __future__ import annotations


def true_positives(T, X, margin=5):
    """Compute true positives without double counting (官方原版逐字移植)."""
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """官方 f_measure：precision 池化（对标注员并集），recall 对标注员宏平均。"""
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def overlap(A, B):
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(S, Sprime):
    """Covering 指标，官方原版实现（逐位对齐 TCPDBench 的 metrics.py）。

    文献出处：Arbeláez, Maire, Fowlkes & Malik, "Contour Detection and
    Hierarchical Image Segmentation", IEEE TPAMI 33(5):898-916, **2011**,
    doi:10.1109/TPAMI.2010.161，式 (8)。

    注：官方 metrics.py 里写作 "Arbaleaz, 2010"，是拼写（Arbeláez）+ 年份（2011）
    双误；此处仅为溯源注释，实现逻辑与官方逐位一致，未作任何改动。
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """对多标注员平均的 covering。"""
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)
