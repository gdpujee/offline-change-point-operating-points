# -*- coding: utf-8 -*-
"""
analyze_matched_k.py — 匹配 k 实验（EXP-MATCHEDK-001）的分析与制表。

NOT_EVIDENCE: Table-B assert literals mirror prose targets (see MAJOR-004);
this file must never count as number provenance (else traces self-certify).

三问：
Q1 固定 k：greedy（BinSeg 顺序）与 optimal（SSE 精确最优 = 罚参 DP 在该 k 的解）谁更好？
Q2 在**各数据集实际被选中的 k** 上比较（消除"选几个"这一混淆，只看"选哪些"）→ 选择规则效应
Q3 F1–k 剖面的形状：TCPD 上 F1 峰在哪个 k？各默认方法实际停在哪个 k？

输出：results/processed/matched_k_table.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

MSEGLEN = 2

# 论文正文「Aggregating (averaging over k = 2..5 within each dataset)」一段所报的值，
# 同时是 EVID-039（§7 功效）的输入。表 D 会逐项断言这些字面量，
# 使「正文数字 ← matched_k.jsonl」成为可执行链条而不是手抄。
KS_AGG = (2, 3, 4, 5)
PAPER_AGG = {"n": 41, "delta_f1": 0.0122, "sd_ddof1": 0.0855,
             "win": 16, "loss": 13, "tie": 12, "wilcoxon_p": 0.456}


def aggregate_k25(per, datasets):
    """k∈KS_AGG 的数据集内平均后的配对差（greedy − optimal）。

    与论文正文同口径：只在四个 k 齐全的数据集上计算（实测 41/41 齐全）。
    """
    g, o = [], []
    for d in datasets:
        gs = [per[d][k]["greedy"]["f1"] for k in KS_AGG
              if k in per[d] and "greedy" in per[d][k]]
        os_ = [per[d][k]["optimal"]["f1"] for k in KS_AGG
               if k in per[d] and "optimal" in per[d][k]]
        if len(gs) != len(KS_AGG) or len(os_) != len(KS_AGG):
            continue
        g.append(float(np.mean(gs)))
        o.append(float(np.mean(os_)))
    return np.array(g) - np.array(o)


def load():
    rows = [json.loads(l) for l in (ROOT / "results/raw/matched_k.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r["minseglen"] == MSEGLEN]
    return rows


def load_tcpd_main():
    rows = json.loads((ROOT / "results/raw/tcpd_main.json").read_text())
    per = defaultdict(dict)
    for r in rows:
        per[r["dataset"]][r["algorithm"]] = r
    return per


def table_b_rows(per, tm, datasets):
    """Q2 rows shared by Table B printing and the derived JSONL mirror.

    Single computation feeding both outputs (zero-divergence by construction):
    per reference method, greedy/optimal F1 at each dataset's actually
    selected k, with means/deltas. Returns list of dicts in ref order.
    """
    rows = []
    for ref in ["binseg", "pelt", "pelt_bic", "segneigh"]:
        g, o = [], []
        for d in datasets:
            if d not in tm or ref not in tm[d]:
                continue
            k = int(round(tm[d][ref]["k"]))
            if k not in per[d] or "greedy" not in per[d][k] or "optimal" not in per[d][k]:
                continue
            g.append(per[d][k]["greedy"]["f1"])
            o.append(per[d][k]["optimal"]["f1"])
        if len(g) < 3:
            continue
        g = np.array(g); o = np.array(o); d = g - o
        w = int((d > 1e-9).sum()); l = int((d < -1e-9).sum()); t = len(d) - w - l
        p = st.wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-12) else 1.0
        rows.append({"ref": ref, "n": len(g),
                     "greedy_mean": float(g.mean()),
                     "optimal_mean": float(o.mean()),
                     "delta_mean": float(d.mean()),
                     "win": w, "loss": l, "tie": t, "wilcoxon_p": float(p)})
    return rows


def main():
    rows = load()
    per = defaultdict(dict)          # ds -> k -> rule -> row
    for r in rows:
        per[r["dataset"]].setdefault(r["k"], {})[r["rule"]] = r
    datasets = sorted(per)
    n_ds = len(datasets)
    ks = sorted({r["k"] for r in rows})

    out = []
    out.append("# 匹配 k 实验（自动生成，勿手改）\n")
    out.append(f"> 数据源 `results/raw/matched_k.jsonl`（EXP-MATCHEDK-001），minseglen={MSEGLEN}，"
               f"n={n_ds} 数据集（与 TCPD 主结果同口径）。\n")
    out.append("> greedy = BinSeg 贪心/瓶颈顺序的前 k 个分裂；"
               "optimal = SSE 精确最优 k 分割（精确 PELT 关于罚参的解路径）。\n")

    # ---- 表 A：固定 k ----
    out.append("\n## 表 A 固定 k 下的 F1（Q1）\n")
    out.append("| k | greedy F1 | optimal F1 | ΔF1 | win | loss | tie | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k in ks:
        g = np.array([per[d][k]["greedy"]["f1"] for d in datasets if k in per[d]])
        o = np.array([per[d][k]["optimal"]["f1"] for d in datasets if k in per[d]])
        if len(g) != len(o) or len(g) == 0:
            continue
        d = g - o
        w = int((d > 1e-9).sum()); l = int((d < -1e-9).sum()); t = len(d) - w - l
        p = st.wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-12) else 1.0
        out.append(f"| {k} | {g.mean():.4f} | {o.mean():.4f} | {d.mean():+.4f} | "
                   f"{w} | {l} | {t} | {p:.4f} |")

    # ---- 表 B：在各自实际选中的 k 上比较（Q2）----
    tm = load_tcpd_main()
    out.append("\n## 表 B 在各数据集**实际选中的 k** 上比较两条选择规则（Q2，消除计数混淆）\n")
    out.append("| 参照方法（决定 k） | n | greedy F1 | optimal F1 | ΔF1 | win | loss | tie | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for _b in table_b_rows(per, tm, datasets):
        ref = _b["ref"]
        out.append(f"| {ref} 的 k | {_b['n']} | {_b['greedy_mean']:.4f} | {_b['optimal_mean']:.4f} | {_b['delta_mean']:+.4f} | "
                   f"{_b['win']} | {_b['loss']} | {_b['tie']} | {_b['wilcoxon_p']:.4f} |")
    # Round-10 MAJOR-004 provenance mirror: the four per-condition deltas as
    # rows (registry min/max over them binds prose +0.003/+0.023). Same
    # variable as Table B above — no second implementation to rot.
    _brows = table_b_rows(per, tm, datasets)
    assert [b["ref"] for b in _brows] == ["binseg", "pelt", "pelt_bic", "segneigh"], \
        "Table B conditions changed"
    assert [b["n"] for b in _brows] == [41, 37, 33, 41], "Table B n changed"
    _dmin = min(b["delta_mean"] for b in _brows)
    _dmax = max(b["delta_mean"] for b in _brows)
    assert round(_dmin, 3) == 0.003, "min condition delta moved: %.6f" % _dmin
    assert round(_dmax, 3) == 0.023, "max condition delta moved: %.6f" % _dmax
    _bpath = ROOT / "results" / "raw" / "derived_matched_deltas.jsonl"
    _bpath.write_text("\n".join(json.dumps(b, sort_keys=True) for b in _brows) + "\n")

    # ---- 表 C：F1–k 剖面 + 各方法实际 k（Q3）----
    out.append("\n## 表 C F1–k 剖面与各默认方法实际停留的 k（Q3）\n")
    out.append("| k | greedy F1 | optimal F1 |")
    out.append("|---|---|---|")
    for k in ks:
        g = np.array([per[d][k]["greedy"]["f1"] for d in datasets if k in per[d]])
        o = np.array([per[d][k]["optimal"]["f1"] for d in datasets if k in per[d]])
        out.append(f"| {k} | {g.mean():.4f} | {o.mean():.4f} |")
    out.append("")
    out.append("| 方法 | 实际平均 k | TCPD F1 |")
    out.append("|---|---|---|")
    for alg in ["zero", "segneigh", "pelt_bic", "sc_pelt", "amoc", "pelt", "binseg"]:
        v = [tm[d][alg] for d in tm if alg in tm[d]]
        out.append(f"| {alg} | {np.mean([x['k'] for x in v]):.2f} | "
                   f"{np.mean([x['f1'] for x in v]):.4f} |")
    # oracle
    orc = []
    for d in datasets:
        best = max((max(per[d][k][r]["f1"] for r in ("greedy", "optimal")))
                   for k in per[d])
        orc.append(best)
    orc_k = []
    for d in datasets:
        bestk, bestv = None, -1
        for k in per[d]:
            v = max(per[d][k][r]["f1"] for r in ("greedy", "optimal"))
            if v > bestv:
                bestv, bestk = v, k
        orc_k.append(bestk)
    out.append(f"| **oracle（k∈0..6 事后最优）** | {np.mean(orc_k):.2f} | {np.mean(orc):.4f} |")

    # ---- 表 D：k=2..5 数据集内平均后的聚合（论文正文 + EVID-039 的输入）----
    # 这是论文正文「Aggregating (averaging over k = 2..5 within each dataset)」一段
    # 与 §7 功效（EVID-039）的 sd 的唯一来源。此前它只在正文里出现、无生成器，
    # 且 analyze_power.py 以手抄字面量 SD=0.0855 / delta=0.0122 引用它。
    diff = aggregate_k25(per, datasets)
    n_a = len(diff)
    delta = float(diff.mean())
    sd1 = float(diff.std(ddof=1))          # 样本 sd（ddof=1）—— 论文 §7 用这个
    sd0 = float(diff.std(ddof=0))          # 总体 sd（ddof=0），仅作对照
    w = int((diff > 1e-9).sum()); l = int((diff < -1e-9).sum()); t = n_a - w - l
    p = float(st.wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0
    out.append("\n## 表 D k=2..5 数据集内平均后的聚合（论文正文 + EVID-039 的输入）\n")
    out.append("> 口径：先在**每个数据集内**对 k∈{2,3,4,5} 取 greedy / optimal 的 F1 平均，"
               "再作差，再跨数据集聚合（minseglen=2，要求该数据集四个 k 齐全）。")
    out.append(f"> `sd` 为**样本 sd（ddof=1）**；论文 §7 引用的 sd=0.0855 即此值"
               f"（总体 sd ddof=0 = {sd0:.4f}，仅供对照）。")
    out.append("")
    out.append("| n | ΔF1 | sd (ddof=1) | win | loss | tie | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|---|")
    out.append(f"| {n_a} | {delta:+.4f} | {sd1:.4f} | {w} | {l} | {t} | {p:.4f} |")
    # 与论文正文 / EVID-039 的硬绑定：字面量一旦抄错，这里立即失败
    assert n_a == PAPER_AGG["n"], f"n {n_a} != {PAPER_AGG['n']}"
    assert round(delta, 4) == PAPER_AGG["delta_f1"], f"ΔF1 {delta:.6f} != {PAPER_AGG['delta_f1']}"
    assert round(sd1, 4) == PAPER_AGG["sd_ddof1"], f"sd {sd1:.6f} != {PAPER_AGG['sd_ddof1']}"
    assert (w, l, t) == (PAPER_AGG["win"], PAPER_AGG["loss"], PAPER_AGG["tie"]), \
        f"win/loss/tie {(w, l, t)} != {(PAPER_AGG['win'], PAPER_AGG['loss'], PAPER_AGG['tie'])}"
    assert round(p, 3) == PAPER_AGG["wilcoxon_p"], f"p {p:.6f} != {PAPER_AGG['wilcoxon_p']}"

    dst = ROOT / "results" / "processed" / "matched_k_table.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nsaved", dst)


if __name__ == "__main__":
    main()
