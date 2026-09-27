# -*- coding: utf-8 -*-
"""
run_algorithm_variants.py — 算法变体对照实验（固化此前一次性脚本）。

变体（全部在 TCPD 37 条上，官方口径 F1/Covering）：
- gap_ratio / gap_diff：贪心分裂强度序列的谱隙截断（k*=argmax d_k/d_{k+1} 或 d_k−d_{k+1}）
- weakest_link：每 k 精确 DP（segneigh 结构）取「最弱分裂 ≥ 3ln n」的最大 k
- weakest_link_mbic：同上，阈值加 MBIC 段长修正 ln m1 + ln m2 − ln m
输出：results/raw/algorithm_variants.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import metrics_tcpdbench as M  # noqa: E402
import tcpd_data as D  # noqa: E402

OUT = ROOT / "results" / "raw" / "algorithm_variants.json"


def greedy_seq(y, K=60):
    """无阈值贪心 binseg（R 候选语义）的分裂强度序列 [(cp, delta), ...]。"""
    ss = L.SumStats(y)
    n = ss.n
    if n < 4:
        return []
    segs = [(0, n)]
    out = []
    for _ in range(K):
        best = None
        for (a, b) in segs:
            lo, hi = max(a + 1, 1), min(b - 2, n - 4)
            if hi < lo:
                continue
            c_arr = np.arange(lo, hi + 1)
            d = ss.sse(a, b) - ss.sse_vec(a, c_arr + 1) - ss.sse_vec(c_arr + 1, b)
            j = int(np.argmax(d))
            if best is None or d[j] > best[0]:
                best = (float(d[j]), int(c_arr[j]))
        if best is None or best[0] <= 0:
            break
        delta, c = best
        out.append((c, delta))
        for i, (a, b) in enumerate(segs):
            if a <= c < b:
                segs[i] = (a, c + 1)
                segs.insert(i + 1, (c + 1, b))
                break
    return out


def gap_rule(y, rule):
    seq = greedy_seq(y)
    if not seq:
        return np.array([], dtype=int)
    d = np.array([x[1] for x in seq])
    if len(d) == 1:
        k = 1 if d[0] > 0 else 0
    elif rule == "ratio":
        k = int(np.argmax(d[:-1] / np.maximum(d[1:], 1e-12))) + 1
    else:
        k = int(np.argmax(d[:-1] - d[1:])) + 1
    return np.array(sorted(x[0] for x in seq[:k]), dtype=int)


def weakest_link(y, beta, mbic=False, K=40):
    ss = L.SumStats(y)
    n = ss.n
    best_k, best_cps = 0, []
    for k in range(1, min(K, n // 2) + 1):
        sg = L.segneigh(y, beta=0.0, Q=k + 1)
        if len(sg) != k:
            break
        b = [0] + [c + 1 for c in sg] + [n]
        ws = [ss.sse(b[i - 1], b[i + 1]) - ss.sse(b[i - 1], b[i]) - ss.sse(b[i], b[i + 1])
              for i in range(1, len(b) - 1)]
        thr = beta
        if mbic:
            a, s, bb = b[0], b[1], b[2]
            m1, m2, m = s - a, bb - s, bb - a
            thr = beta + np.log(m1) + np.log(m2) - np.log(m)
        w = min(ws) if ws else np.inf
        if w >= thr:
            best_k, best_cps = k, sg
        else:
            break
    return np.array(best_cps, dtype=int)


def main():
    rows = []
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except FileNotFoundError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        ann = D.load_annotations(ds)
        variants = {
            "gap_ratio": gap_rule(y, "ratio"),
            "gap_diff": gap_rule(y, "diff"),
            "weakest_link": weakest_link(y, 3.0 * np.log(n), mbic=False),
            "weakest_link_mbic": weakest_link(y, 3.0 * np.log(n), mbic=True),
        }
        for name, cps in variants.items():
            f1, p, r = M.f_measure(ann, cps.tolist(), return_PR=True)
            cov = M.covering(ann, cps.tolist(), n)
            rows.append({"dataset": ds, "algorithm": name, "k": len(cps),
                         "cps": cps.tolist(), "f1": f1, "precision": p,
                         "recall": r, "covering": cov})
        print(ds, "done", flush=True)

    OUT.write_text(json.dumps(rows))
    print(f"saved {len(rows)} rows -> {OUT}")

    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r["algorithm"]].append(r)
    for a, v in agg.items():
        print(f"{a:20s} F1={np.mean([r['f1'] for r in v]):.4f} "
              f"Cov={np.mean([r['covering'] for r in v]):.4f} k={np.mean([r['k'] for r in v]):.2f}")


if __name__ == "__main__":
    main()
