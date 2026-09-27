# -*- coding: utf-8 -*-
"""
run_uncapped.py — B3 重跑：用真正无上限（Q = n//2）替换 Q=200，
解决 review_round3 B3 指出的"+192 是被截断值"问题。

仅跑 σ=10 单元（此处 Q=200 撞上限），保留 n∈{1000,10000}，k∈{0,5,20}，3 reps；
对照组另加 gauss σ=1（Q=200 远不撞上限，应与原结果一致）。
输出 results/raw/uncapped.jsonl。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import synthetic as S  # noqa: E402

OUT = ROOT / "results" / "raw" / "uncapped.jsonl"
NS = [1000, 10000]
KS = [0, 5, 20]
NOISES = ["gauss", "t3", "ar1_09"]
AMPS = [1.0, 3.0]
SIGMAS = [1.0, 10.0]
REPS = 3


def pr_metrics(true_cps, pred_cps, n):
    tol = max(5, int(round(0.01 * n)))
    used = [False] * len(pred_cps)
    tp = 0
    for t in true_cps:
        best = None
        for i, g in enumerate(pred_cps):
            if not used[i] and abs(t - g) <= tol:
                if best is None or abs(t - pred_cps[best]) > abs(t - g):
                    best = i
        if best is not None:
            used[best] = True
            tp += 1
    rec = tp / len(true_cps) if true_cps else 1.0
    prec = sum(used) / len(pred_cps) if pred_cps else 1.0
    return prec, rec


def binseg_floor(y, Q):
    n = len(y)
    sigma2 = L.mad_sigma(y) ** 2
    pen = max(3.0 * np.log(n), 2.0 * np.log(n) * sigma2)
    return L.binseg_r(y, pen=pen, Q=Q)


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], r["rep"], r["algorithm"]))
    rows_out = open(OUT, "a")
    computed = 0

    def emit(row):
        rows_out.write(json.dumps(row) + "\n")
        rows_out.flush()

    cells = [(n, k, nz, a, s) for n in NS for k in KS for nz in NOISES for a in AMPS for s in SIGMAS]
    for ci, (n, k, nz, a, s) in enumerate(cells, 1):
        for rep in range(REPS):
            seed = ci * 100 + rep
            y, true_cps = S.gen_series(n, k, nz, a, s, seed=seed)
            logn = np.log(n)
            q_inf = max(k + 1, n // 2)  # 真正无上限（n//2 远大于所有合理 k）
            # 同 run_qcap.py：续跑键必须与写出的 dataset 一致，否则重跑会重复追加。
            dsname = f"syn_{n}_{k}_{nz}_a{a}_s{s}_{rep}"
            for name, fn in (
                ("binseg_qinf", lambda y=y: L.binseg_r(y, pen=3.0 * logn, Q=q_inf)),
                ("binseg_qinf_floor", lambda y=y: binseg_floor(y, q_inf)),
            ):
                if (dsname, rep, name) in done:
                    continue
                t0 = time.perf_counter()
                cps = fn(y)
                dt = time.perf_counter() - t0
                p, r = pr_metrics(true_cps, list(map(int, cps)), n)
                emit({"dataset": dsname, "rep": rep,
                      "algorithm": name, "k": len(cps), "k_true": len(true_cps),
                      "precision": p, "recall": r, "runtime_s": dt,
                      "n": n, "k_true_cell": k, "noise": nz, "sigma": s})
                computed += 1
        print(f"cell {ci}/{len(cells)} done", flush=True)
    rows_out.close()
    print("computed=%d skipped=%d (total %d)"
          % (computed, len(done), computed + len(done)))
    if computed == 0 and len(done) > 0:
        # Round-11 MAJOR-006: full resume-hit is REUSED, not recomputed-PASS.
        # rc 2 = reused-noop (cf. make_manuscript_zip.py no-TeX rc 2); the
        # cold-start driver maps it to REUSED status. Partial resume
        # (computed > 0) still exits 0: real work was done.
        print("REUSED: all %d rows already present, nothing recomputed"
              % len(done))
        return 2

    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    from collections import defaultdict
    print("\n=== 网格均值（按 noise × σ，与表 6 口径一致）===")
    print(f"{'algo':22s} {'noise/sig':12s} {'kerr':>9s} {'max_kpred':>10s} {'n_rep':>6s}")
    for nz in NOISES:
        for s in SIGMAS:
            for a in ("binseg_qinf", "binseg_qinf_floor"):
                sel = [r for r in rows if r["algorithm"] == a
                       and r["noise"] == nz and r["sigma"] == s]
                if not sel:
                    continue
                kerrs = [r["k"] - r["k_true"] for r in sel]
                print(f"{a:22s} {nz:8s}/σ{int(s):<3d} {np.mean(kerrs):+9.2f} "
                      f"{int(np.max([r['k'] for r in sel])):>10d} {len(sel):>6d}")
    print("\n=== 按 k_true 分层（σ=10 关键单元）===")
    print(f"{'algo':22s} {'noise':8s} {'ktrue':>5s} {'kerr':>9s} {'max_kpred':>10s} {'n':>3s}")
    for nz in NOISES:
        for kt in KS:
            for a in ("binseg_qinf", "binseg_qinf_floor"):
                sel = [r for r in rows if r["algorithm"] == a and r["noise"] == nz
                       and r["sigma"] == 10.0 and r["k_true_cell"] == kt]
                if not sel:
                    continue
                kerrs = [r["k"] - r["k_true"] for r in sel]
                print(f"{a:22s} {nz:8s} {kt:5d} {np.mean(kerrs):+9.2f} "
                      f"{int(np.max([r['k'] for r in sel])):>10d} {len(sel):>3d}")


if __name__ == "__main__":
    sys.exit(main())
