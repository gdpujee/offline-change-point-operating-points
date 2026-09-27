# -*- coding: utf-8 -*-
"""
run_scale.py — 规模扩展实验 v2（增量保存，防中途丢失）。

变更（v1 教训：一次性 JSON 只在结尾写出，中断即全丢）：
- 每行立即追加写入 results/raw/scale.jsonl（JSON Lines）
- n=10^6：sc_v3/bic_var 只跑 1 rep（Python 常数因子大，见 FAILURES F4）；
  mbic_raw/binseg_raw 跑 3 reps（快）
- n=10^4/10^5：全部 3 reps
gauss σ=1, k=5, amp=3。
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import synthetic as S  # noqa: E402

OUT = ROOT / "results" / "raw" / "scale.jsonl"


def pr(true_cps, pred_cps, n):
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


def timed_with_mem(fn, y):
    """计时（不使用 tracemalloc——其实测开销 2-4×，见 FAILURES F4b）。

    内存按结构估计：所有算法 O(n) 工作数组；进程 RSS 增量仅供参考。
    """
    import resource
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    t0 = time.perf_counter()
    cps = fn(y)
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return cps, dt, max(rss1 - rss0, 0.0)


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["n"], r["rep"], r["algorithm"]))
        print(f"resuming, {len(done)} rows already present")

    computed = 0
    with open(OUT, "a") as f:
        for n, reps_full, reps_slow in [(10_000, 3, 3), (100_000, 3, 3), (1_000_000, 3, 1)]:
            for rep in range(max(reps_full, reps_slow)):
                seed = rep
                y, true_cps = S.gen_series(n, 5, "gauss", 3.0, 1.0, seed=seed)
                algos = {
                    "mbic_raw": (3, lambda y: L.pelt_mbic(y, pen=3.0 * np.log(len(y)))),
                    "binseg_raw": (3, lambda y: L.binseg_r(y, pen=3.0 * np.log(len(y)), Q=5)),
                    "bic_var": (reps_slow if n == 1_000_000 else 3,
                                lambda y: L.pelt(y, beta=2.0 * np.log(len(y)) * np.var(y))),
                    "sc_v3": (reps_slow if n == 1_000_000 else 3,
                              lambda y: L.sc_pelt(y, seed=seed, use_floor=True)),
                }
                for name, (reps, fn) in algos.items():
                    if rep >= reps:
                        continue
                    if (n, rep, name) in done:
                        continue
                    cps, dt, mem = timed_with_mem(fn, y)
                    p, r = pr(true_cps, list(map(int, cps)), n)
                    row = {"n": n, "rep": rep, "algorithm": name, "k": len(cps),
                           "k_true": 5, "precision": p, "recall": r,
                           "runtime_s": dt, "peak_mem_mb": mem}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"n={n} rep={rep} {name}: {dt:.2f}s k={len(cps)}", flush=True)
                    computed += 1
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
    agg = defaultdict(list)
    for r in rows:
        agg[(r["algorithm"], r["n"])].append(r)
    print(f"\n{'algorithm':11s} {'n':>9s} {'runtime_s':>10s} {'mem_MB':>8s} {'k':>6s} {'P':>6s} {'R':>6s}")
    for (a, n), v in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{a:11s} {n:9d} {np.mean([r['runtime_s'] for r in v]):10.3f} "
              f"{np.mean([r['peak_mem_mb'] for r in v]):8.1f} "
              f"{np.mean([r['k'] for r in v]):6.2f} "
              f"{np.mean([r['precision'] for r in v]):6.3f} "
              f"{np.mean([r['recall'] for r in v]):6.3f}")


if __name__ == "__main__":
    sys.exit(main())
