# -*- coding: utf-8 -*-
"""
run_tcpd_main.py — TCPD 主实验（E1）+ 机理原始数据（M1/M2 依据）。

每个数据集 × 每个算法：运行 → CP 集合（0-based）→ 指标（官方口径 F1/Covering，
含 return_PR 的 P/R）→ 运行时间。SC-PELT 跑 N_SEEDS 个独立种子。

算法：
  amoc/binseg/pelt/segneigh/zero —— R changepoint MBIC/BIC 语义忠实复现（已 100% 对拍）
  amoc_nophantom/binseg_nophantom/... —— 已默认去幻影（本文件输出均不含幻影）
  binseg_textbook —— 教科书贪心+阈值停止（对照用，机理分析）
  pelt_bic —— 标准常数罚参 PELT，β=2ln n（经典文献口径）
  sc_pelt —— 本文方法（置换校准 + 自洽 Bonferroni + PELT）

输出：results/raw/tcpd_main.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import metrics_tcpdbench as M  # noqa: E402
import tcpd_data as D  # noqa: E402

N_SEEDS = 20
# 41 = the 42 BENCH_DATASETS entries minus uk_coal_employ (non-finite values),
# and it is the dataset count every table in the paper is built on. Asserted,
# not assumed: a silent partial run used to overwrite the shipped
# results/raw/tcpd_main.json with an empty file (R58-1).
EXPECTED_DATASETS = 41
OUT = ROOT / "results" / "raw" / "tcpd_main.json"

# 教科书式 binseg（贪心 + 阈值停止，无瓶颈规则、无 Q 上限缺省 5）——机理对照
def binseg_textbook(y, beta):
    ss = L.SumStats(y)
    n = ss.n
    cps = []
    segs = [(0, n)]
    while True:
        best = None
        for (a, b) in segs:
            if b - a < 2:
                continue
            lo, hi = a + 1, b - 2
            if hi < lo:
                continue
            c_arr = np.arange(lo, hi + 1)
            d = ss.sse(a, b) - ss.sse_vec(a, c_arr + 1) - ss.sse_vec(c_arr + 1, b)
            j = int(np.argmax(d))
            if best is None or d[j] > best[0]:
                best = (float(d[j]), int(c_arr[j]))
        if best is None or best[0] < beta:
            break
        delta, c = best
        cps.append(c)
        for i, (a, b) in enumerate(segs):
            if a <= c < b:
                segs[i] = (a, c + 1)
                segs.insert(i + 1, (c + 1, b))
                break
    return np.array(sorted(cps), dtype=int)


def score(ann, pred, n):
    f1, p, r = M.f_measure(ann, pred, margin=5, return_PR=True)
    cov = M.covering(ann, pred, n)
    return {"f1": f1, "precision": p, "recall": r, "covering": cov}


def main():
    rows = []
    loaded: list[str] = []
    nonfinite: list[str] = []
    absent: list[str] = []
    for ds in D.BENCH_DATASETS:
        try:
            y, n_obs = D.load_series(ds)
        except D.MissingDataError:
            absent.append(ds)
            continue
        if not np.all(np.isfinite(y)):
            nonfinite.append(ds)
            continue
        loaded.append(ds)
        ann = D.load_annotations(ds)
        logn = np.log(n_obs)

        def timed(fn, *a, **kw):
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            dt = time.perf_counter() - t0
            return out, dt

        algos = {
            "amoc": lambda: L.amoc(y, pen=3.0 * logn),
            "binseg": lambda: L.binseg_r(y, pen=3.0 * logn, Q=5),
            "pelt": lambda: L.pelt_mbic(y, pen=3.0 * logn),
            "segneigh": lambda: L.segneigh(y, beta=2.0 * logn, Q=5),
            "zero": lambda: L.zero(y),
            "binseg_textbook": lambda: binseg_textbook(y, beta=3.0 * logn),
            "pelt_bic": lambda: L.pelt(y, beta=2.0 * logn),
        }
        for name, fn in algos.items():
            cps, dt = timed(fn)
            s = score(ann, cps.tolist(), n_obs)
            rows.append({"dataset": ds, "algorithm": name, "seed": None,
                         "k": len(cps), "runtime_s": dt, "cps": cps.tolist(), **s})

        # SC-PELT：20 个独立种子
        for seed in range(N_SEEDS):
            cps, dt = timed(lambda: L.sc_pelt(y, seed=seed))
            s = score(ann, cps.tolist(), n_obs)
            rows.append({"dataset": ds, "algorithm": "sc_pelt", "seed": seed,
                         "k": len(cps), "runtime_s": dt, "cps": cps.tolist(), **s})
        print(f"{ds:22s} done", flush=True)

    if not rows:
        raise SystemExit(
            f"FATAL: 0 rows -- no TCPD series could be loaded "
            f"(absent: {len(absent)}/{len(D.BENCH_DATASETS)}). "
            f"Refusing to overwrite {OUT}. "
            f"Run: python3 experiments/check_data.py")
    if len(loaded) != EXPECTED_DATASETS:
        raise SystemExit(
            f"FATAL: loaded {len(loaded)} datasets, expected "
            f"{EXPECTED_DATASETS} (absent={absent}, non-finite={nonfinite}). "
            f"Refusing to write a partial {OUT}.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    print(f"saved {len(rows)} rows -> {OUT} "
          f"({len(loaded)} datasets; non-finite skipped: {nonfinite})")


if __name__ == "__main__":
    main()
