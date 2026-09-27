# -*- coding: utf-8 -*-
"""
analyze_paired_vs_binseg.py — 复算论文 `tab:paired`（"All paired Wilcoxon tests
against binseg, n=41"）的全部 7 行，把它从"手抄数字"变成可重跑的命令。

为什么有这个脚本
----------------
Round 17 的三遍核验（AT-57）发现：`tab:paired` 的 7 行**没有任何生成器**。

  - `analyze_paired_tests.py` 只算 2 个检验（binseg vs pelt、floor vs q5），
    不覆盖 pelt_bic / segneigh / amoc / zero / sc_pelt / binseg_textbook；
  - `analyze_stats.py` 算的是 sc_pelt vs 各 baseline（另一套对比），且写的是
    `results/processed/stats_tcpd.csv`，不是这张表；
  - 该表的实际出处是 `notes/review_round3.md` §B1 的一张表，其复算脚本自述为
    "一次性内联脚本"，**未入库** ⇒ 审稿人无法从仓库复算这张表。

本脚本补上生成器。复算结果同时**订正**了 round3 表里两个被"双重舍入"的 p 值
（详见下方"订正"一节）。

输入（已入库，只读）
--------------------
- `results/raw/tcpd_main.json` → 8 个算法在 41 条 TCPD 数据集上的逐条 F1

方法（必须写明，否则不可复算）
------------------------------
- 逐数据集配对：`diffs[i] = f1(binseg, ds_i) - f1(alg, ds_i)`，n=41；
- `scipy.stats.wilcoxon(diffs)`：scipy 默认 `method='auto'`、`zero_method='wilcox'`、双侧；
- win/loss/tie 用 `TIE_TOL = 1e-9` 判定（与 `analyze_stats.py` / `analyze_paired_tests.py` 一致）；
- 表内 p 值按论文体例给到 3 位小数，故断言 `round(p, 3)` 逐行相等。

sc_pelt 的聚合口径（本脚本的核心断言）
--------------------------------------
`sc_pelt` 是唯一一个每个数据集有 **20 条**记录（20 个随机种子）的算法，其余算法
每条数据集只有 1 条。因此"取均值"还是"取单次"会影响秩检验的并列结构。

本脚本**断言 20 条记录逐位相同**（`len(set(reps)) == 1`），从而把 EVID-020 的
"20 个种子输出完全一致"从文字声明变成可执行断言；既然逐位相同，取均值与取任一次
在**精确算术**下等价，本脚本直接取该公共值。

⚠️ 注意：`np.mean(20 个相同浮点数)` **不是恒等映射** —— 逐对求和的中间舍入会使
23/41 个数据集的均值偏离该值 ≤1 ULP，而这足以打破 5 个真实并列
（centralia / gdp_croatia / gdp_japan / ratner_stock / usd_isk），把 p 从 0.190804
推到 0.196696。即 `analyze_stats.py` 用 `np.mean` 得到的 sc_pelt 行 p 值带有这层
伪影。本表用单次值，得 **0.191**，与论文一致。

订正（论文原印 → 本表）
----------------------
三处差异都是"先舍到 4 位小数、再舍到 3 位"的**双重舍入**产物（或等价的
"两个已舍入量相减"）：

    pelt_bic 的 p : 真值 0.227476 →(4dp) 0.2275 →(3dp) 0.228   原印 0.228，正确 0.227
    zero     的 p : 真值 0.028455 →(4dp) 0.0285 →(3dp) 0.029   原印 0.029，正确 0.028
    sc_pelt 的ΔF1 : 真值 0.045485 ；等价路径 0.7270−0.6815=0.0455 →(3dp) 0.046
                                                               原印 +0.046，正确 +0.045

单步舍入才是论文自述口径（p 与 ΔF1 均给 3 位小数），故本表断言 `round(值, 3)`。
三处订正**都不改变任何结论**：两个 p 一个仍非显著、一个仍在名义显著区间
（zero 0.028 < 0.05）；ΔF1 的 0.001 位移不影响"BinSeg 均值领先但逐数据集不可
分辨"这一核心陈述。正文"仅 segneigh/amoc/zero 名义显著（0.045/0.048/0.029）"
需同步把 0.029 改为 0.028。

输出
----
- 控制台：逐行表格 + 可直接粘贴的 LaTeX 行
- `results/raw/paired_vs_binseg.json`

⚠️ 本文件**故意不写入输入文件的 SHA-256**。`tcpd_main.json` 含墙钟 `runtime_s`，
其哈希每跑一次上游就会变；若把该哈希写进本文件，就必须给判据再加一条 class-D 豁免。
按 Round 15 的教训（豁免必须"声明式 + 按文件作用域 + 配正反向负控"，且裸前缀规则是
fail-open），不为一个仅作留痕的字段去扩大豁免面。输入的版本由**提交 revision** 锚定，
输入哈希记录在 `review/REVIEW_STATE.md` 的 R18 条目里。
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import scipy.stats as st

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
SRC = RAW / "tcpd_main.json"
OUT = RAW / "paired_vs_binseg.json"

TIE_TOL = 1e-9
BASELINE = "binseg"

# 论文 `tab:paired` 的行顺序。每项：
#   (算法, 正确 ΔF1@3dp, win/loss/tie, 正确 p@3dp, 论文原印 ΔF1@3dp, 论文原印 p@3dp)
#
# ΔF1 与 p 都按论文体例只给 3 位小数，故断言 round(值, 3) —— 这正是读者能核对的精度。
# "原印"两列只用于打印订正说明；断言一律对"正确"列。
EXPECTED = [
    ("pelt",             0.016, (8, 7, 26),   0.691, 0.016, 0.691),
    ("pelt_bic",         0.044, (17, 11, 13), 0.227, 0.044, 0.228),  # p 双重舍入
    ("segneigh",         0.052, (21, 7, 13),  0.045, 0.052, 0.045),
    ("amoc",             0.047, (18, 5, 18),  0.048, 0.047, 0.048),
    ("zero",             0.073, (28, 11, 2),  0.028, 0.073, 0.029),  # p 双重舍入
    ("sc_pelt",          0.045, (18, 11, 12), 0.191, 0.046, 0.191),  # ΔF1 双重舍入
    ("binseg_textbook",  0.005, (4, 3, 34),   0.735, 0.005, 0.735),  # w/l/t 首次固化
]


def load_vectors():
    rows = json.loads(SRC.read_text())
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["dataset"]][r["algorithm"]].append(r["f1"])

    datasets = sorted(per)
    algs = sorted({a for d in per.values() for a in d})

    # --- EVID-020：sc_pelt 的 20 个种子必须逐位相同 ---
    n_reps = {len(per[ds]["sc_pelt"]) for ds in datasets}
    assert n_reps == {20}, f"expected 20 sc_pelt reps per dataset, got {sorted(n_reps)}"
    not_bit_identical = [ds for ds in datasets if len(set(per[ds]["sc_pelt"])) != 1]
    assert not not_bit_identical, (
        "EVID-020 violated: sc_pelt seeds are not bit-identical on "
        f"{not_bit_identical}")
    print(f"[EVID-020] sc_pelt: 41/41 datasets have 20 bit-identical seed records — "
          f"determinism asserted, not assumed.")

    vec = {}
    for a in algs:
        if a == "sc_pelt":
            # 逐位相同 ⇒ 公共值即精确均值；不用 np.mean（它会引入 ≤1 ULP 伪影）
            vec[a] = np.array([per[ds][a][0] for ds in datasets])
        else:
            missing = [ds for ds in datasets if len(per[ds][a]) != 1]
            assert not missing, f"{a}: expected 1 record per dataset, missing {missing}"
            vec[a] = np.array([per[ds][a][0] for ds in datasets])
    return datasets, vec


def paired(diffs):
    n = int(len(diffs))
    win = int(np.sum(diffs > TIE_TOL))
    loss = int(np.sum(diffs < -TIE_TOL))
    tie = n - win - loss
    p = float(st.wilcoxon(diffs).pvalue)
    return n, float(diffs.mean()), (win, loss, tie), p


def main():
    datasets, vec = load_vectors()
    assert len(datasets) == 41, f"expected n=41, got {len(datasets)}"
    base = vec[BASELINE]

    print(f"\nsource : {SRC.relative_to(ROOT)}")
    print(f"env    : python {'.'.join(map(str, __import__('sys').version_info[:3]))}  "
          f"numpy {np.__version__}  scipy {scipy.__version__}")
    print(f"n      : {len(datasets)}   baseline: {BASELINE}\n")

    hdr = (f"{'comparison':30s} {'dF1':>8s} {'win/loss/tie':>14s} {'p':>9s}"
           f"   {'printed':>13s}   note")
    print(hdr)
    print("-" * len(hdr))

    out_rows, latex, n_checked = [], [], 0
    n_corrected = 0
    for alg, exp_df1, exp_wlt, exp_p, pr_df1, pr_p in EXPECTED:
        d = base - vec[alg]          # binseg - baseline
        n, mean, wlt, p = paired(d)

        if round(mean, 3) != exp_df1:
            raise AssertionError(f"{alg}: dF1 {round(mean,3)} != expected {exp_df1}")
        if wlt != exp_wlt:
            raise AssertionError(f"{alg}: w/l/t {wlt} != expected {exp_wlt}")
        if round(p, 3) != exp_p:
            raise AssertionError(f"{alg}: p {round(p,3)} != expected {exp_p}")

        notes = []
        if pr_df1 != exp_df1:
            notes.append(f"dF1 {pr_df1:+.3f}->{exp_df1:+.3f}")
        if pr_p != exp_p:
            notes.append(f"p {pr_p:.3f}->{exp_p:.3f}")
        if notes:
            n_corrected += 1
        note = ("CORRECTED: " + ", ".join(notes)) if notes else ""
        n_checked += 1

        print(f"{BASELINE} vs {alg:22s} {mean:+8.4f} {wlt[0]:>5d}/{wlt[1]}/{wlt[2]:<6d} "
              f"{p:9.6f}   {pr_df1:+.3f}/{pr_p:.3f}   {note}")
        out_rows.append({
            "comparison": f"{BASELINE} vs {alg}",
            "algorithm": alg,
            "n": n,
            "mean_delta_f1": mean,
            "mean_delta_f1_3dp": round(mean, 3),
            "win": wlt[0], "loss": wlt[1], "tie": wlt[2],
            "wilcoxon_p": p,
            "wilcoxon_p_3dp": round(p, 3),
            "printed_delta_f1_3dp": pr_df1,
            "printed_p_3dp": pr_p,
            "corrected": bool(notes),
        })
        label = alg.replace("_", r"\_")
        latex.append(f"binseg vs {label:<22s} & {mean:+.3f} & {round(p, 3):.3f} \\\\")

    print(f"\n{n_checked}/{len(EXPECTED)} rows recomputed and asserted "
          f"(n, win/loss/tie, dF1@3dp, p@3dp).")
    print(f"{n_corrected} row(s) carry a correction against the previously printed value; "
          f"no conclusion changes (all three are last-digit renderings).")

    payload = {
        "what": "tab:paired — all paired Wilcoxon tests against binseg (n=41)",
        "source": str(SRC.relative_to(ROOT)),
        "method": ("scipy.stats.wilcoxon(diffs): method='auto', zero_method='wilcox', "
                   "two-sided; diffs[i] = f1(binseg, ds_i) - f1(alg, ds_i); "
                   "win/loss/tie threshold TIE_TOL=1e-9. "
                   "sc_pelt uses its single seed value: its 20 seed records are "
                   "bit-identical on all 41 datasets (asserted here; EVID-020)."),
        "n_datasets": len(datasets),
        "baseline": BASELINE,
        "rows": out_rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"\nsaved  : {OUT.relative_to(ROOT)}")
    print("\nLaTeX-formatted rows for the paired comparison table:")
    for l in latex:
        print("  " + l)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
