# -*- coding: utf-8 -*-
"""check_crossversion_ulp.py — 跨 Python 版本的 1-ULP 漂移：机械测量 + 论文声称断言。

为什么要有这个脚本
------------------
`REPRODUCIBILITY_AUDIT.md` §5.7.3d 与 README 用**散文**描述了「哪些指标、哪些数据集、
多少条」在 Python <3.12 与 >=3.12 之间会差 1 ULP。散文里的事实没人能一键复算，
且 Round 22 实测发现其中一句（「`covering` 在这 41 条上恰好未受影响」）**是假的**。

本脚本把该结论变成可执行检查：

  # 只 dump 当前环境的逐位结果
  python3 experiments/check_crossversion_ulp.py

  # 与另一个解释器对比，并断言论文 Appendix B 的声称
  python3 experiments/check_crossversion_ulp.py --other /path/to/other/python

判据说明（重要）
----------------
- 逐位相同用 **hex 字符串全等**判断，**不用 `np.std`**（`np.std(20 个相同浮点数)`
  会报 3.3e-16 的自身求和伪影；见 skill §66.14）。
- 「是否影响报告值」按论文**实际印刷精度**逐档比较（2/3/4/6 dp），不设统一绝对容差
  （统一容差会造出满屏假 FAIL；见 skill §66.14 同源教训）。

论文声称（Appendix B 第 3 条，main.tex）由 `PAPER_*` 常量固化，改错即失败。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import metrics_tcpdbench as M  # noqa: E402
import tcpd_data as D  # noqa: E402

COLS = ("f1", "precision", "recall", "covering")

# --- 论文 Appendix B 固化声称（main.tex "Floating-point metrics depend on the
# --- Python interpreter version"）-------------------------------------------
PAPER_F1_BINSEG = {
    "apple", "bee_waggle_6", "businv", "centralia", "gdp_iran",
    "scanline_126007", "seatbelts",
}
PAPER_F1_PELT = {
    "apple", "bee_waggle_6", "businv", "centralia", "gdp_iran",
    "occupancy", "seatbelts", "well_log",
}
PAPER_PRECISION_UNAFFECTED = True

# Round 22 实测：covering **并**受影响（§5.7.3d 原写「恰好未受影响」，为假）
COVERING_BINSEG_MEASURED = {
    "bitcoin", "brent_spot", "children_per_woman", "construction",
    "gdp_japan", "rail_lines", "ratner_stock",
}


def dump():
    """重算 results/raw/tcpd_main.json 里每条记录的四项指标，返回 hex 行列表。"""
    rows = json.loads((ROOT / "results" / "raw" / "tcpd_main.json").read_text())
    ann_cache, n_cache = {}, {}

    def n_obs(name):
        if name not in n_cache:
            p = D.DATA_ROOT / name / f"{name}.json"
            n_cache[name] = int(json.loads(p.read_text())["n_obs"])
        return n_cache[name]

    out = []
    for i, r in enumerate(rows):
        name = r["dataset"]
        if name not in ann_cache:
            ann_cache[name] = D.load_annotations(name)
        ann = ann_cache[name]
        F, P, Rc = M.f_measure(ann, r["cps"], return_PR=True)
        C = M.covering(ann, r["cps"], n_obs(name))
        out.append([i, name, r["algorithm"], r["seed"],
                    F.hex(), P.hex(), Rc.hex(), C.hex()])
    return out


def run_other(python):
    r = subprocess.run([python, str(Path(__file__).resolve()), "--dump-json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"other interpreter failed:\n{r.stderr[-800:]}")
    return json.loads(r.stdout)


def affected(rows_a, rows_b, algo, col):
    """(algo, col) 上逐位不同的**数据集**集合。"""
    j = 4 + COLS.index(col)
    ds = set()
    for a, b in zip(rows_a, rows_b):
        if a[2] != algo:
            continue
        if a[j] != b[j]:
            ds.add(a[1])
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", help="另一个 Python 解释器路径（与本环境对比）")
    ap.add_argument("--dump-json", action="store_true", help="输出 JSON 后退出")
    args = ap.parse_args()

    if args.dump_json:
        json.dump(dump(), sys.stdout)
        sys.stdout.write("\n")
        return

    mine = dump()
    if not args.other:
        print(f"{sys.version.split()[0]}: {len(mine)} records recomputed "
              f"(use --other <python> to compare and assert)")
        return

    other = run_other(args.other)
    assert len(mine) == len(other), f"record count {len(mine)} != {len(other)}"

    n_diff = sum(1 for a, b in zip(mine, other) if a != b)
    print(f"bit-differing rows: {n_diff} / {len(mine)}")

    # --- 按算法 × 指标的影响面 ---
    per = defaultdict(Counter)
    for a, b in zip(mine, other):
        if a == b:
            continue
        for j, c in enumerate(COLS):
            if a[4 + j] != b[4 + j]:
                per[a[2]][c] += 1
    print(f"\n{'algorithm':<20} " + " ".join(f"{c:>9}" for c in COLS))
    print("-" * 60)
    for algo in sorted(per):
        print(f"{algo:<20} " + " ".join(f"{per[algo][c]:>9}" for c in COLS))

    # --- 报告值是否在论文印刷精度下仍然改变 ---
    print()
    for ndp in (2, 3, 4, 6):
        bad = 0
        for a, b in zip(mine, other):
            if a == b:
                continue
            for j in range(4):
                va, vb = float.fromhex(a[4 + j]), float.fromhex(b[4 + j])
                if round(va, ndp) != round(vb, ndp):
                    bad += 1
        print(f"  differ at {ndp} dp: {bad}")

    # --- 断言论文声称 ---
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and cond

    print("\nassertions (paper claims, Appendix B):")
    check(affected(mine, other, "binseg", "f1") == PAPER_F1_BINSEG,
          f"binseg f1 affected set == paper's 7 "
          f"(got {sorted(affected(mine, other, 'binseg', 'f1'))})")
    check(affected(mine, other, "pelt", "f1") == PAPER_F1_PELT,
          f"pelt f1 affected set == paper's 8 "
          f"(got {sorted(affected(mine, other, 'pelt', 'f1'))})")
    prec = set()
    for algo in sorted(per):
        prec |= affected(mine, other, algo, "precision")
    check(not prec and PAPER_PRECISION_UNAFFECTED,
          f"precision unaffected everywhere (got {sorted(prec)})")

    bad4 = 0
    for a, b in zip(mine, other):
        if a == b:
            continue
        for j in range(4):
            if round(float.fromhex(a[4 + j]), 4) != round(float.fromhex(b[4 + j]), 4):
                bad4 += 1
    check(bad4 == 0, f"no reported value changes at 4 dp (got {bad4})")

    cov = affected(mine, other, "binseg", "covering")
    check(cov == COVERING_BINSEG_MEASURED,
          f"binseg covering **is** affected on 7 series (Round 22 correction; "
          f"got {sorted(cov)})")

    print()
    if not ok:
        raise SystemExit("cross-version ULP claims do not reproduce - investigate")
    print("ALL CLAIMS REPRODUCED")


if __name__ == "__main__":
    main()
