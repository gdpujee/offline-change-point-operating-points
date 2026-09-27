# -*- coding: utf-8 -*-
"""
analyze_paired_tests.py — 复算论文里报告的两个配对 Wilcoxon 检验（EVID-031），
并固化"这两个 p 值为何跨环境稳定、而其它同类量不稳定"的判决证据。

为什么有这个脚本
----------------
论文正文与投稿信报告了两个 p 值：

  ① binseg vs pelt，逐数据集配对（n=41）：ΔF1=+0.0164、win/loss/tie=8/7/26、p=0.691
     —— 支撑"BinSeg 胜 PELT 在数据上不可分辨"这一核心重述（EVID-031）。
  ② UBG-floor vs binseg_q5，逐数据集配对（n=41）：ΔF1=+0.000707、仅 1 条变化、p=0.317
     —— 支撑"去上限在 TCPD 上几乎无影响"（EVID-031 / C5）。

`EVIDENCE_LEDGER.md` 的 EVID-031 行把处理脚本写成 `run_qcap.py` / `analyze_matched_k.py`，
但**这两个脚本都不计算上述两个 p 值**（`analyze_stats.py` 算的是 sc_pelt vs 各 baseline）。
即：论文的头条 p 值此前**没有可执行的生成器**，只存在于台账文字里。
本脚本把这两个数字固化为可重跑的命令，与 `analyze_power.py`（EVID-039 的落地）同一思路。

输入（全部已入库，只读）
------------------------
- `results/raw/tcpd_main.json`       → binseg（Q=5）与 pelt 的逐数据集 F1
- `results/raw/qcap.jsonl`           → 41 条 TCPD 上的 binseg_q5 / binseg_q200 / binseg_q200_floor

输出
----
- 控制台摘要（含"环境指纹"与"鲁棒性"两节，这两节**只打印、不写文件**，
  因为它们的取值依赖运行环境，写进 JSON 会给本文件引入不可复现面）
- `results/raw/paired_tests.json`    → 输入 SHA-256 + 方法 + 全部输出（供审计）

方法（必须写明，否则不可复算）
------------------------------
`scipy.stats.wilcoxon(diffs)`，即 scipy 默认 `method='auto'`（本数据下选到正态近似路径，
因为存在并列的零差）、`zero_method='wilcox'`、双侧。差值为 `A - B`，逐数据集一一配对。

`scipy` 的 `WilcoxonResult` **不暴露所选 method**，因此本脚本用显式 `method='approx'`
/ `method='exact'` 复算并做**逐位比对**来确定 auto 实际走了哪条路径，而不是凭文档断言。

跨环境稳定性（本脚本附带的核心结论）
------------------------------------
本脚本复算的两个 p 值（0.690945 / 0.317311）在
  env A = Python 3.13.12 + numpy 2.5.3 + scipy 1.18.1
  env B = Python 3.10.12 + numpy 2.2.6 + scipy 1.15.3
下**逐位相同**，且与论文报告值一致。

但这不是普遍性质，根因是 **CPython 3.12 改了浮点 `sum()` 的实现**
（朴素累加 → Neumaier 补偿求和），因此任何 `sum(list_of_floats)` 的取值都依赖 Python 版本：

    terms = [1.0, 1.0, 1.0, 1/3, 1.0]        # apple 的 5 个标注员比率，两环境下逐位相同
    py3.13: sum(terms) = 4.333333333333333   （= math.fsum）
    py3.10: sum(terms) = 4.333333333333334   （≠ math.fsum）

本仓自研代码里只有两处浮点 `sum()`，都在 `scripts/metrics_tcpdbench.py`：
  - 第 49 行  `R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) ...)`  → recall
  - 第 107 行 `return sum(Cs) / len(Cs)`                        → covering
（precision 只用整除除法，故跨版本稳定；各脚本里的 `sum(used)` 是对 bool 求和，
  是精确整数运算，同样稳定。上游 `data/TCPDBench/analysis/scripts/metrics.py` 是同两个
  表达式，本仓是忠实移植，因此这是**上游继承**的性质而非移植错误。）

后果：`recall`（进而 `f1`）在 41 条 TCPD 数据集里有 7 条出现 ≤1 ULP 差异，使得
  (a) 已入库的 `tcpd_main.json` 与 `qcap.jsonl` 在"同一算法、同一 margin"下互不一致；
  (b) 论文第二个 p 值随之在 0.317311 / 0.773688 之间跳（见本脚本 robustness 一节）；
  (c) `results/processed/stats_tcpd.csv` 的 `wilcoxon_p` 列被放大到 5.1e-2（§5.7.3c）。
详见 `REPRODUCIBILITY_AUDIT.md` §5.7.3d。
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
RAW = ROOT / "results" / "raw"
OUT = RAW / "paired_tests.json"

TIE_TOL = 1e-9  # 判定 win/loss/tie 的阈值（与 analyze_stats.py 一致）

# 论文 / 投稿信报告值（用于断言复现）
PAPER_1 = {"n": 41, "mean_delta_f1": 0.0164, "win": 8, "loss": 7, "tie": 26,
           "wilcoxon_p": 0.691, "median_binseg": 0.7386, "median_pelt": 0.7536}
PAPER_2 = {"n": 41, "mean_delta_f1": 0.000707, "n_changed": 1, "wilcoxon_p": 0.317}

# p 值断言容差：论文只给到 3 位小数（0.691 / 0.317），故允许 5e-4
P_TOL = 5e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(diffs: np.ndarray) -> dict:
    """复算一个配对 Wilcoxon 检验，并用显式 method 复算来确证 auto 走了哪条路径。

    返回 dict（键名即写入 JSON 的字段名）。
    """
    n = int(len(diffs))
    n_nonzero = int(np.count_nonzero(diffs))
    win = int(np.sum(diffs > TIE_TOL))
    loss = int(np.sum(diffs < -TIE_TOL))
    tie = n - win - loss
    mean = float(diffs.mean())

    base = {"n": n, "n_nonzero_diffs": n_nonzero, "mean_delta_f1": mean,
            "win": win, "loss": loss, "tie": tie}

    if n_nonzero == 0:
        base.update(wilcoxon_p=1.0, wilcoxon_statistic=None,
                    wilcoxon_method_used="all-zero diffs (p set to 1 by convention)")
        return base

    res = st.wilcoxon(diffs)  # method='auto', zero_method='wilcox', two-sided
    p = float(res.pvalue)
    stat = float(res.statistic)

    cand = {}
    for name in ("approx", "exact"):
        try:
            cand[name] = float(st.wilcoxon(diffs, method=name).pvalue)
        except Exception as exc:  # exact 在并列差上可能直接报错
            cand[name] = None

    matched = [k for k, v in cand.items() if v is not None and v == p]
    detail = ", ".join(
        f"{k}={v!r}" if v is not None else f"{k}=N/A" for k, v in cand.items()
    )
    base.update(
        wilcoxon_p=p,
        wilcoxon_statistic=stat,
        wilcoxon_method_used=(
            "scipy.stats.wilcoxon(diffs): method='auto' -> "
            f"{matched[0] if len(matched) == 1 else matched} "
            f"(verified bit-for-bit against explicit re-runs: {detail}); "
            "zero_method='wilcox', two-sided"
        ),
    )
    return base


def load() -> tuple:
    main_path = RAW / "tcpd_main.json"
    qcap_path = RAW / "qcap.jsonl"
    main_rows = json.loads(main_path.read_text())
    qcap_rows = [json.loads(l) for l in qcap_path.read_text().splitlines() if l.strip()]
    return main_path, qcap_path, main_rows, qcap_rows


def vec_main(main_rows, alg):
    m = {}
    for r in main_rows:
        if r["algorithm"] == alg:
            m.setdefault(r["dataset"], []).append(r["f1"])
    return {d: float(np.mean(v)) for d, v in m.items()}


def vec_qcap(qcap_rows, alg, keep):
    m = {}
    for r in qcap_rows:
        if r["algorithm"] == alg and r["dataset"] in keep:
            m.setdefault(r["dataset"], []).append(r["f1"])
    return {d: float(np.mean(v)) for d, v in m.items()}


def fingerprint(main_rows, qcap_rows, tcpd_datasets):
    """自洽性指纹：用 tcpd_main 自带的 cps 重算指标，看它复现哪一个已入库文件。

    这一节**只打印**。返回值是 dict，但只用于控制台；写文件时不带环境相关字段。
    """
    try:
        import metrics_tcpdbench as M
        import tcpd_data as D
    except Exception as exc:  # data/ 缺失或 import 失败
        print(f"[指纹] 跳过（无法导入指标/数据模块：{type(exc).__name__}: {exc}）")
        return None

    B = {r["dataset"]: r for r in main_rows if r["algorithm"] == "binseg"}
    Q5 = vec_qcap(qcap_rows, "binseg_q5", tcpd_datasets)
    common = sorted(set(B) & set(Q5))

    hit_main = hit_qcap = hit_none = 0
    for ds in common:
        try:
            ann = D.load_annotations(ds)
            _, n = D.load_series(ds)
        except (FileNotFoundError, D.MissingDataError):
            print("[指纹] 跳过（data/ 下缺数据集，无法重算）")
            return None
        f1, _, _ = M.f_measure(ann, B[ds]["cps"], margin=5, return_PR=True)
        hm, hq = (f1 == B[ds]["f1"]), (f1 == Q5[ds])
        hit_main += hm
        hit_qcap += hq
        hit_none += (not hm and not hq)

    f1_diff = [d for d in common if B[d]["f1"] != Q5[d]]
    print(f"[指纹] 用 tcpd_main 自带 cps + 当前 metrics_tcpdbench 重算 f1：")
    print(f"       命中 tcpd_main.json {hit_main}/{len(common)}  "
          f"命中 qcap.jsonl {hit_qcap}/{len(common)}  都不命中 {hit_none}/{len(common)}")
    print(f"       两文件 f1 不同的数据集 {len(f1_diff)} 条：{f1_diff}")
    print(f"       -> 本环境与 tcpd_main.json 的生成环境同类"
          if hit_main > hit_qcap else
          f"       -> 本环境与 qcap.jsonl 的生成环境同类")
    return {"n_common": len(common), "hit_main": hit_main, "hit_qcap": hit_qcap,
            "f1_diff_datasets": f1_diff}


def root_cause_probe():
    """打印 CPython `sum()` 版本差异的最小复现（只打印）。"""
    terms = [1.0, 1.0, 1.0, 1 / 3, 1.0]
    s, f = sum(terms), math.fsum(terms)
    print(f"[根因] terms={terms}")
    print(f"       sum(terms)={s!r}   math.fsum(terms)={f!r}   "
          f"{'相同' if s == f else '不同 -> 本环境是 Python <3.12 的朴素累加'}")
    return {"sum": repr(s), "fsum": repr(f), "agree": s == f}


def main() -> int:
    main_path, qcap_path, main_rows, qcap_rows = load()
    tcpd_datasets = {r["dataset"] for r in main_rows}

    # ---- ① binseg vs pelt，逐数据集配对（n=41）------------------------------
    b, pe = vec_main(main_rows, "binseg"), vec_main(main_rows, "pelt")
    ds1 = sorted(set(b) & set(pe))
    d1 = np.array([b[d] - pe[d] for d in ds1])
    r1 = paired(d1)

    # ---- ② UBG-floor vs binseg_q5，逐数据集配对（n=41，同在 qcap.jsonl 内）---
    fl = vec_qcap(qcap_rows, "binseg_q200_floor", tcpd_datasets)
    q5 = vec_qcap(qcap_rows, "binseg_q5", tcpd_datasets)
    ds2 = sorted(set(fl) & set(q5))
    d2 = np.array([fl[d] - q5[d] for d in ds2])
    r2 = paired(d2)

    # ---- ②' 鲁棒性：把对照臂换成"另一个文件里的同一算法" ----------------------
    # tcpd_main.json 的 binseg 与 qcap.jsonl 的 binseg_q5 是同一次调用
    # （L.binseg_r(y, pen=3.0*logn, Q=5)），但两文件的 recall 浮点路径不同
    # （见 §5.7.3d），于是配对差从 1 个非零变成 8 个，p 从 0.317 跳到 0.774。
    ds2b = sorted(set(fl) & set(b))
    d2b = np.array([fl[d] - b[d] for d in ds2b])
    r2b = paired(d2b)

    # ---- ③ 恒等性自检：TCPD 上 binseg_q200 vs binseg_q200_floor 的 k ---------
    ks = {}
    for r in qcap_rows:
        if r["dataset"] in tcpd_datasets and r["algorithm"] in ("binseg_q200", "binseg_q200_floor"):
            ks[(r["dataset"], r["algorithm"])] = r["k"]
    ds3 = sorted({d for d, a in ks if (d, "binseg_q200") in ks and (d, "binseg_q200_floor") in ks})
    same_k = sum(1 for d in ds3 if ks[(d, "binseg_q200")] == ks[(d, "binseg_q200_floor")])

    # ---- ③b 恒等性自检 2：binseg_q5 vs binseg_q200_floor 的 k（EVID-031 的第二条主张）----
    # 这条直接解释了检验②为何只有 1 条非零差：Q=5 与 floor 版只在 occupancy 上分道。
    kq = {}
    for r in qcap_rows:
        if r["dataset"] in tcpd_datasets and r["algorithm"] in ("binseg_q5", "binseg_q200_floor"):
            kq[(r["dataset"], r["algorithm"])] = r
    ds4 = sorted({d for d, a in kq if (d, "binseg_q5") in kq and (d, "binseg_q200_floor") in kq})
    diff4 = [d for d in ds4 if kq[(d, "binseg_q5")]["k"] != kq[(d, "binseg_q200_floor")]["k"]]
    same_k2 = len(ds4) - len(diff4)

    # ---- 断言：论文报告值必须复现 -------------------------------------------
    checks = [
        ("① p", abs(r1["wilcoxon_p"] - PAPER_1["wilcoxon_p"]) < P_TOL, r1["wilcoxon_p"]),
        ("① ΔF1", abs(r1["mean_delta_f1"] - PAPER_1["mean_delta_f1"]) < 5e-5, r1["mean_delta_f1"]),
        ("① win/loss/tie",
         (r1["win"], r1["loss"], r1["tie"]) == (PAPER_1["win"], PAPER_1["loss"], PAPER_1["tie"]),
         (r1["win"], r1["loss"], r1["tie"])),
        ("② p", abs(r2["wilcoxon_p"] - PAPER_2["wilcoxon_p"]) < P_TOL, r2["wilcoxon_p"]),
        ("② ΔF1", abs(r2["mean_delta_f1"] - PAPER_2["mean_delta_f1"]) < 5e-6, r2["mean_delta_f1"]),
        ("② n_changed", r2["n_nonzero_diffs"] == PAPER_2["n_changed"], r2["n_nonzero_diffs"]),
        ("③ k 恒等", same_k == len(ds3) and len(ds3) == 41, f"{same_k}/{len(ds3)}"),
        ("③b k 恒等(q5 vs floor)",
         same_k2 == 40 and len(ds4) == 41 and diff4 == ["occupancy"],
         f"{same_k2}/{len(ds4)}，差异={diff4}"),
    ]
    ok = all(c[1] for c in checks)

    out = {
        "script": "experiments/analyze_paired_tests.py",
        "evidence_id": "EVID-031",
        "method": ("scipy.stats.wilcoxon(diffs), default method='auto' (verified by explicit "
                   "re-run to resolve to the normal approximation here, because of tied zero "
                   "differences), zero_method='wilcox', two-sided; diff = A - B, paired "
                   "one-to-one per dataset; win/loss/tie threshold = 1e-9"),
        "inputs": {
            "results/raw/tcpd_main.json": sha256(main_path),
            "results/raw/qcap.jsonl": sha256(qcap_path),
        },
        "test_1_binseg_vs_pelt": {
            **r1,
            "median_binseg": float(np.median([b[d] for d in ds1])),
            "median_pelt": float(np.median([pe[d] for d in ds1])),
            "paper_reports": PAPER_1,
        },
        "test_2_floor_vs_q5": {**r2, "paper_reports": PAPER_2},
        "robustness_test_2_other_file": {
            **r2b,
            "note": ("同一算法、同一 margin=5，但对照臂取自 tcpd_main.json 的 binseg 而非 "
                     "qcap.jsonl 的 binseg_q5。两文件的 recall 浮点路径不同（CPython sum() "
                     "语义差异，见 REPRODUCIBILITY_AUDIT.md §5.7.3d），使配对差从 1 个非零 "
                     "变成 8 个，Wilcoxon p 由 0.317311 跳到 0.773688。"
                     "论文用的是**同文件内配对**（floor 与 q5 都取自 qcap.jsonl），"
                     "这是正确口径；本条记录的是该口径对外部复核者的敏感性。"
                     "注意本条的 win/loss/tie=1/0/40 与 n_nonzero_diffs=8 并不矛盾，"
                     "且正是脆弱性的来源：win/loss/tie 用 1e-9 容差（故 7 个 ≤1 ULP 差"
                     "被算作 tie），而 scipy 的 zero_method='wilcox' 只丢弃**精确零**，"
                     "这 7 个差值仍然进入秩统计、改变有效样本量（1 -> 8）从而改变 p。"),
        },
        "identity_check": {
            "note": "TCPD 上 binseg_q200 vs binseg_q200_floor 的 k 逐条相同（地板在 z-scored 数据上恒不触发）",
            "n_datasets": len(ds3),
            "n_identical_k": same_k,
        },
        "identity_check_2": {
            "note": ("TCPD 上 binseg_q5 vs binseg_q200_floor 的 k：除 occupancy 外逐条相同。"
                     "这条直接解释了检验②为何只有 1 条非零差。"),
            "n_datasets": len(ds4),
            "n_identical_k": same_k2,
            "differing": [
                {"dataset": d,
                 "k_binseg_q5": kq[(d, "binseg_q5")]["k"],
                 "k_binseg_q200_floor": kq[(d, "binseg_q200_floor")]["k"],
                 "f1_binseg_q5": kq[(d, "binseg_q5")]["f1"],
                 "f1_binseg_q200_floor": kq[(d, "binseg_q200_floor")]["f1"]}
                for d in diff4
            ],
        },
        "root_cause": {
            "note": ("CPython 3.12 起浮点 sum() 改为 Neumaier 补偿求和，故任何 sum(float list) "
                     "的取值依赖 Python 版本（≤1 ULP）。本仓自研代码只有两处："
                     "scripts/metrics_tcpdbench.py 第 49 行（recall）与第 107 行（covering）。"),
            "minimal_repro": "terms=[1.0,1.0,1.0,1/3,1.0]; py3.13 sum=4.333333333333333, py3.10 sum=4.333333333333334",
            "reported_values_affected": False,
            "reported_values_note": "论文按 4 位小数报告，≤1 ULP 不影响任何已报告数字",
        },
    }

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    print(f"python {sys.version.split()[0]} / numpy {np.__version__} / scipy {scipy.__version__}")
    print("=" * 96)
    print("① binseg vs pelt（逐数据集配对，n=%d，非零差 %d）"
          % (r1["n"], r1["n_nonzero_diffs"]))
    print(f"   ΔF1={r1['mean_delta_f1']:+.6f}  win/loss/tie={r1['win']}/{r1['loss']}/{r1['tie']}  "
          f"median binseg/pelt={out['test_1_binseg_vs_pelt']['median_binseg']:.4f}/"
          f"{out['test_1_binseg_vs_pelt']['median_pelt']:.4f}")
    print(f"   Wilcoxon p={r1['wilcoxon_p']:.6f}  stat={r1['wilcoxon_statistic']}")
    print(f"   {r1['wilcoxon_method_used']}")
    print(f"   论文报告：ΔF1=+0.0164  win/loss/tie=8/7/26  p=0.691  median=0.7386/0.7536")
    print("=" * 96)
    print("② UBG-floor vs binseg_q5（逐数据集配对，n=%d，非零差 %d）"
          % (r2["n"], r2["n_nonzero_diffs"]))
    print(f"   ΔF1={r2['mean_delta_f1']:+.6f}  win/loss/tie={r2['win']}/{r2['loss']}/{r2['tie']}")
    print(f"   Wilcoxon p={r2['wilcoxon_p']:.6f}  stat={r2['wilcoxon_statistic']}")
    print(f"   {r2['wilcoxon_method_used']}")
    print(f"   论文报告：ΔF1=+0.000707  1 条变化  p=0.317")
    print("=" * 96)
    print("②' 鲁棒性：对照臂换成 tcpd_main.json 的 binseg（同一算法，另一文件）")
    print(f"   n={r2b['n']}  非零差 {r2b['n_nonzero_diffs']}  "
          f"ΔF1={r2b['mean_delta_f1']:+.6f}  Wilcoxon p={r2b['wilcoxon_p']:.6f}")
    print("=" * 96)
    print(f"③ 恒等自检：binseg_q200 vs binseg_q200_floor 的 k 在 TCPD 上 "
          f"{same_k}/{len(ds3)} 相同")
    print(f"③b 恒等自检：binseg_q5 vs binseg_q200_floor 的 k 在 TCPD 上 "
          f"{same_k2}/{len(ds4)} 相同，差异={diff4}")
    for d in diff4:
        print(f"      {d}: k {kq[(d, 'binseg_q5')]['k']} -> "
              f"{kq[(d, 'binseg_q200_floor')]['k']}  "
              f"F1 {kq[(d, 'binseg_q5')]['f1']:.4f} -> "
              f"{kq[(d, 'binseg_q200_floor')]['f1']:.4f}")
    print("=" * 96)
    fingerprint(main_rows, qcap_rows, tcpd_datasets)
    print("=" * 96)
    root_cause_probe()
    print("=" * 96)
    for name, good, val in checks:
        print(f"   [{'OK  ' if good else 'FAIL'}] {name:<16} = {val}")
    print(f"saved {OUT.relative_to(ROOT)}")
    if not ok:
        print("!! 论文报告值未复现，请调查 !!", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
