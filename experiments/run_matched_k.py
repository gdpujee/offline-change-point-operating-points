# -*- coding: utf-8 -*-
"""
run_matched_k.py — 决定性实验：在**匹配变点数 k** 的条件下比较两条选择规则。

动机（review_round3 B1）：TCPD 上 binseg vs pelt 的 ΔF1=+0.0164 且 Wilcoxon p=0.691，
无法主张"BinSeg 更好"。但两者预测的变点数不同（binseg k=2.00 vs pelt k=2.66 vs
pelt_bic k=3.90），所以"更好"可能只是**计数差异**，而非**选择规则差异**。

本实验把计数固定住：对每个 k，比较
  (a) greedy  ：BinSeg 的贪心/瓶颈顺序下的前 k 个分裂（= BinSeg 在恰好 k 个变点时的输出）
  (b) optimal ：SSE 精确最优的 k 变点分割（= 罚参 DP / PELT 在对应罚参下会选的解，
                因为精确 PELT 关于 β 的解路径恰为最优 k 分割路径）
若同 k 下 (a) 仍优于 (b) → 差异来自**选择规则**（支持 max-min 机理）；
若同 k 下无差异      → 差异只来自**计数**，机理表述必须改为"计数规则"。

输出：results/raw/matched_k.jsonl（增量）
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import metrics_tcpdbench as M  # noqa: E402
import tcpd_data as D  # noqa: E402

OUT = ROOT / "results" / "raw" / "matched_k.jsonl"
KS = list(range(0, 7))
MSEGLEN = 2  # 与 BinSeg 候选约束（左段 ≥2 点）对齐；另跑 minseglen=1 作稳健性


def optimal_k_segmentations(y, kmax, minseglen=1):
    """精确 DP：对每个 k ∈ 0..kmax 返回总 SSE 最小的 k 变点分割（0-based 变点）。

    Best[k][j] = 用 k 个变点分割 y[0:j] 的最小 SSE。段长 ≥ minseglen。
    """
    ss = L.SumStats(y)
    n = ss.n
    kmax = min(kmax, (n // minseglen) - 1)
    Best = np.full((kmax + 1, n + 1), np.inf)
    par = np.full((kmax + 1, n + 1), -1, dtype=np.int64)
    Best[0, 0] = 0.0
    for j in range(minseglen, n + 1):
        Best[0, j] = ss.sse(0, j)
    for q in range(1, kmax + 1):
        for j in range((q + 1) * minseglen, n + 1):
            lo = q * minseglen
            hi = j - minseglen
            if hi < lo:
                continue
            v = np.arange(lo, hi + 1)
            vals = Best[q - 1, v] + ss.sse_vec(v, j)
            i = int(np.argmin(vals))
            Best[q, j] = vals[i]
            par[q, j] = v[i]
    out = {}
    for q in range(0, kmax + 1):
        if not np.isfinite(Best[q, n]):
            continue
        cps, j, qq = [], n, q
        while qq > 0:
            v = par[qq, j]
            if v <= 0:
                break
            cps.append(v - 1)  # 变点 = 左段末点（0-based）
            j = v
            qq -= 1
        out[q] = np.array(sorted(cps), dtype=int)
    return out


def binseg_greedy_order(y, Q):
    """BinSeg 的贪心分裂顺序（与 cpd_lib.binseg_r 完全相同的候选/argmax 语义，
    但不施加接受阈值），返回按贪心顺序排列的 0-based 变点列表。"""
    ss = L.SumStats(y)
    n = ss.n
    if n < 4:
        return []
    segs = [(0, n)]
    order = []
    for _ in range(Q):
        best = None
        for (a, b) in segs:
            lo = max(a + 1, 1)
            hi = min(b - 2, n - 4)
            if hi < lo:
                continue
            c_arr = np.arange(lo, hi + 1)
            m = b - a
            m1 = c_arr + 1 - a
            m2 = b - (c_arr + 1)
            sse_full = ss.sse(a, b)
            d = sse_full - ss.sse_vec(a, c_arr + 1) - ss.sse_vec(c_arr + 1, b)
            lam_c = 0.5 * d - 0.5 * (np.log(m1) + np.log(m2) - np.log(m))
            j = int(np.argmax(lam_c))
            if best is None or lam_c[j] > best[0]:
                best = (float(lam_c[j]), int(c_arr[j]))
        if best is None:
            break
        _, c = best
        order.append(c)
        for i, (a, b) in enumerate(segs):
            if a <= c < b:
                segs[i] = (a, c + 1)
                segs.insert(i + 1, (c + 1, b))
                break
    return order


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], r["k"], r["rule"], r["minseglen"]))
    fout = open(OUT, "a")

    def emit(row):
        fout.write(json.dumps(row) + "\n")
        fout.flush()

    datasets = []
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except (FileNotFoundError, OSError):
            continue
        # 与 run_tcpd_main.py 口径一致：只剔除含非有限值的序列（uk_coal_employ）→ 41 条。
        # centralia 仅 n=15，仍保留以保证与主结果同一样本。
        if not np.all(np.isfinite(y)):
            continue
        datasets.append((ds, y, n, D.load_annotations(ds)))
    print(f"datasets: {len(datasets)}", flush=True)

    for di, (ds, y, n, ann) in enumerate(datasets, 1):
        for msl in [MSEGLEN, 1]:
            opt = optimal_k_segmentations(y, max(KS), minseglen=msl)
            order = binseg_greedy_order(y, max(KS) + 1)
            for k in KS:
                if k not in opt:
                    continue
                for rule, cps in (("greedy", np.array(sorted(order[:k]), dtype=int)),
                                  ("optimal", opt[k])):
                    if (ds, k, rule, msl) in done:
                        continue
                    f1, p, r = M.f_measure(ann, cps.tolist(), return_PR=True)
                    cov = M.covering(ann, cps.tolist(), n)
                    emit({"dataset": ds, "n": n, "k": k, "rule": rule, "minseglen": msl,
                          "f1": f1, "covering": cov, "precision": p, "recall": r,
                          "k_pred": len(cps)})
        print(f"[{di}/{len(datasets)}] {ds} done", flush=True)
    fout.close()

    # ---- 汇总 ----
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["minseglen"], r["k"])][r["rule"]].append(r)
    print("\n=== 匹配 k 下的 F1（41 数据集均值）===")
    print(f"{'msl':>4} {'k':>3} {'greedy':>8} {'optimal':>8} {'Δ':>8} {'win':>4} {'loss':>4} {'tie':>4} {'p':>9}")
    for msl in [MSEGLEN, 1]:
        for k in KS:
            if (msl, k) not in agg:
                continue
            g = agg[(msl, k)]["greedy"]
            o = agg[(msl, k)]["optimal"]
            if not g or not o or len(g) != len(o):
                continue
            fg = np.array([x["f1"] for x in g])
            fo = np.array([x["f1"] for x in o])
            d = fg - fo
            w = int((d > 1e-9).sum())
            l = int((d < -1e-9).sum())
            t = len(d) - w - l
            p = st.wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-12) else 1.0
            print(f"{msl:>4} {k:>3} {fg.mean():8.4f} {fo.mean():8.4f} {d.mean():+8.4f} "
                  f"{w:>4} {l:>4} {t:>4} {p:9.4f}")
        print()

    # 以数据集为单位的聚合配对检验（把 k 作为重复测量在数据集内先取平均）
    for msl in [MSEGLEN, 1]:
        per = defaultdict(dict)
        for (m, k), rules in agg.items():
            if m != msl:
                continue
            for rule in ("greedy", "optimal"):
                for x in rules[rule]:
                    per[x["dataset"]].setdefault(rule, {})[k] = x["f1"]
        for lo, hi, tag in [(1, 6, "k=1..6"), (2, 5, "k=2..5"), (0, 6, "k=0..6")]:
            d = []
            for ds0, v in per.items():
                ks = sorted(set(v["greedy"]) & set(v["optimal"]))
                ks = [k for k in ks if lo <= k <= hi]
                if len(ks) != hi - lo + 1:
                    continue
                d.append(np.mean([v["greedy"][k] - v["optimal"][k] for k in ks]))
            d = np.array(d)
            if len(d) < 3:
                continue
            w = int((d > 1e-9).sum())
            l = int((d < -1e-9).sum())
            t = len(d) - w - l
            p = st.wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-12) else 1.0
            print(f"[msl={msl}] 数据集内先对 {tag} 取平均后配对（n={len(d)}）："
                  f" ΔF1={d.mean():+.4f} win/loss/tie={w}/{l}/{t} Wilcoxon p={p:.4f}")
        # F1-k 剖面（用于定位各规则的最优 k）
        print(f"[msl={msl}] F1-k 剖面：", end=" ")
        for k in KS:
            if (msl, k) not in agg:
                continue
            g = np.mean([x["f1"] for x in agg[(msl, k)]["greedy"]])
            o = np.mean([x["f1"] for x in agg[(msl, k)]["optimal"]])
            print(f"k{k}: G{g:.3f}/O{o:.3f}", end="  ")
        print("\n")


if __name__ == "__main__":
    main()
