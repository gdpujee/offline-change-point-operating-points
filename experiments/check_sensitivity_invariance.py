# -*- coding: utf-8 -*-
"""check_sensitivity_invariance.py — SC-PELT 敏感性网格（EVID-020）的机械复核。

为什么要有这个脚本
------------------
`EVIDENCE_LEDGER.md` / `CLAIMS_MATRIX.md` / `RESEARCH_LOG.md` 三处把 EVID-020 写成
「对 B∈{30,100,300} × α∈{0.01,0.05} × L∈{自适应, n/10, 8} **全部不敏感**」，
读起来是 3×2×3 = 18 个组合。而 `experiments/run_sensitivity.py:29-34` 的网格实际是
**6 + 2 = 8 个配置**（B×α 全组合取 L=None，再只对 B=100/α=0.01 加 L="n10" 与 L=8）。

「不敏感」这个结论本身**实测成立且比声称更强**（见下），但**网格描述与实际不符** ——
正是 skill §66.17 第 2 条：census 必须与实际网格**双向一致**，不能把笛卡尔积当成
实际跑过的网格。

本脚本把 EVID-020 变成可执行检查：

  python3 experiments/check_sensitivity_invariance.py

判据说明
--------
- 逐位相同用 **hex 全等**判断，不用 `np.std`（见 skill §66.14：判据自身也会舍入）。
- 报告值按论文印刷精度比较（F1 4 位、k 2 位）。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- 实际网格（run_sensitivity.py:29-34；**不是** 3x2x3=18）---------------
EXPECTED_CONFIGS = {
    "B30_a0.01_LNone", "B30_a0.05_LNone",
    "B100_a0.01_LNone", "B100_a0.05_LNone",
    "B300_a0.01_LNone", "B300_a0.05_LNone",
    "B100_a0.01_Ln10", "B100_a0.01_L8",
}
PAPER_F1_FLOOR = 0.6815        # tab:sens，F1 (floor=True)，4 dp
PAPER_K_FLOOR = 3.95           # tab:sens，mean k (floor=True)，2 dp
PAPER_F1_NOFLOOR_RANGE = (0.47, 0.52)   # fig:sens 题注 "F1 ~ 0.47--0.52"
PAPER_K_NOFLOOR_RANGE = (35.0, 42.0)    # fig:sens 题注 "mean k ~ 35--42"


def f1_of(r):
    p, rc = r["precision"], r["recall"]
    if p + rc == 0:
        return 0.0
    return p * rc / (0.5 * rc + 0.5 * p)


def main():
    rows = json.loads((ROOT / "results" / "raw" / "sensitivity.json").read_text())
    tcpd = [r for r in rows if r["data"].startswith("tcpd_")]
    cfgs = sorted({r["config"] for r in tcpd})

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print(f"TCPD records: {len(tcpd)}   configs: {len(cfgs)}")
    check(set(cfgs) == EXPECTED_CONFIGS,
          f"grid is the actual 8 configs, not 3x2x3=18 "
          f"(got {len(cfgs)}: {cfgs})")

    for floor, tag in ((True, "floor=True"), (False, "floor=False")):
        sel = [r for r in tcpd if r["floor"] is floor]
        per = {}
        for c in cfgs:
            rs = [r for r in sel if r["config"] == c]
            per[c] = (sum(f1_of(r) for r in rs) / len(rs),
                      sum(r["k_pred"] for r in rs) / len(rs))
        print(f"\n{tag}: mean F1 / mean k per config")
        for c in cfgs:
            print(f"  {c:<20} F1={per[c][0]:.10f}  k={per[c][1]:.6f}")

    # --- floor=True：论文声称「constant across the entire grid」 ---
    sel = [r for r in tcpd if r["floor"] is True]
    means = [sum(f1_of(r) for r in sel if r["config"] == c) /
             sum(1 for r in sel if r["config"] == c) for c in cfgs]
    kmeans = [sum(r["k_pred"] for r in sel if r["config"] == c) /
              sum(1 for r in sel if r["config"] == c) for c in cfgs]
    print()
    check(len({round(m, 4) for m in means}) == 1 and round(means[0], 4) == PAPER_F1_FLOOR,
          f"floor=True: mean F1 identical at 4 dp == {PAPER_F1_FLOOR} "
          f"(got {sorted({round(m, 4) for m in means})})")
    check(len({round(k, 2) for k in kmeans}) == 1 and round(kmeans[0], 2) == PAPER_K_FLOOR,
          f"floor=True: mean k identical at 2 dp == {PAPER_K_FLOOR} "
          f"(got {sorted({round(k, 2) for k in kmeans})})")

    # 比论文更强的判据：逐数据集、逐位相同
    by_ds = defaultdict(dict)
    for r in sel:
        by_ds[r["data"]][r["config"]] = (f1_of(r), r["k_pred"])
    full = [ds for ds, m in by_ds.items() if len(m) == len(cfgs)]
    bit_f1 = sum(1 for ds in full if len({v[0].hex() for v in by_ds[ds].values()}) == 1)
    same_k = sum(1 for ds in full if len({v[1] for v in by_ds[ds].values()}) == 1)
    check(len(full) == 41, f"all 41 datasets carry all {len(cfgs)} configs "
                           f"(got {len(full)})")
    check(bit_f1 == 41, f"floor=True: F1 BIT-identical across configs on all 41 "
                        f"datasets (got {bit_f1})")
    check(same_k == 41, f"floor=True: k identical across configs on all 41 "
                        f"datasets (got {same_k})")

    # --- floor=False：论文给的范围 ---
    sel = [r for r in tcpd if r["floor"] is False]
    means = [sum(f1_of(r) for r in sel if r["config"] == c) /
             sum(1 for r in sel if r["config"] == c) for c in cfgs]
    kmeans = [sum(r["k_pred"] for r in sel if r["config"] == c) /
              sum(1 for r in sel if r["config"] == c) for c in cfgs]
    lo, hi = PAPER_F1_NOFLOOR_RANGE
    klo, khi = PAPER_K_NOFLOOR_RANGE
    # 论文 fig:sens 题注写的是 "F1 ~ 0.47--0.52" / "mean k ~ 35--42"，即**印刷精度**
    # （F1 2 dp、k 整数）。判据必须按同一精度比，否则 0.4683 会被自己的判据
    # 误判为越界（skill §66.14：判据的精度单位必须与论文印刷精度一致）。
    f1r = sorted({round(m, 2) for m in means})
    kr = sorted({round(k) for k in kmeans})
    check(all(lo <= round(m, 2) <= hi for m in means),
          f"floor=False: mean F1 within {PAPER_F1_NOFLOOR_RANGE} at 2 dp "
          f"(got {min(f1r)}..{max(f1r)})")
    check(all(klo <= round(k) <= khi for k in kmeans),
          f"floor=False: mean k within {PAPER_K_NOFLOOR_RANGE} at 0 dp "
          f"(got {min(kr)}..{max(kr)})")
    check(len({round(m, 4) for m in means}) > 1,
          "floor=False: F1 genuinely varies (the contrast is real)")

    print()
    if not ok:
        raise SystemExit("EVID-020 sensitivity claims do not reproduce - investigate")
    print("ALL CLAIMS REPRODUCED")


if __name__ == "__main__":
    main()
