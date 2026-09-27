# -*- coding: utf-8 -*-
"""
run_sensitivity.py — SC-PELT 参数敏感性（B, alpha, L, pilot_mult）+ floor 消融。

设计（skill §31/§32）：
- 消融：floor on/off（sc_v3 vs sc_nofloor）
- B ∈ {30, 100, 300}；alpha ∈ {0.01, 0.05}；L ∈ {自适应, n//10, 8}
- 数据：合成 4 个代表 cell（gauss σ=10、t3 σ=10、ar1_09 σ=10，k=5 amp=3，n=5000，
  3 reps）+ TCPD 全部 37 条（fixed seed 0）
输出：results/raw/sensitivity.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import metrics_tcpdbench as M  # noqa: E402
import synthetic as S  # noqa: E402
import tcpd_data as D  # noqa: E402

OUT = ROOT / "results" / "raw" / "sensitivity.json"

CONFIGS = []
for B in [30, 100, 300]:
    for alpha in [0.01, 0.05]:
        CONFIGS.append({"B": B, "alpha": alpha, "L": None})
CONFIGS.append({"B": 100, "alpha": 0.01, "L": "n10"})
CONFIGS.append({"B": 100, "alpha": 0.01, "L": 8})


def run_config(y, n, cfg, seed, use_floor):
    Lblk = cfg["L"]
    if Lblk == "n10":
        Lblk = max(2, n // 10)
    cps, info = L.sc_pelt(y, B=cfg["B"], alpha=cfg["alpha"], L=Lblk,
                          seed=seed, use_floor=use_floor, return_info=True)
    return cps, info


def main():
    rows = []
    # --- 合成部分 ---
    cells = [(5000, 5, "gauss", 3.0, 10.0), (5000, 5, "t3", 3.0, 10.0),
             (5000, 5, "ar1_09", 3.0, 10.0), (5000, 5, "gauss", 3.0, 1.0)]
    for ci, (n, k, nz, a, s) in enumerate(cells):
        for rep in range(3):
            # 不要用 hash()：Python 3.11 起 str 的哈希由 SipHash24 改为 SipHash13，
            # 即使 PYTHONHASHSEED=0，跨 Python 版本的 hash 值也不同（实测 py3.10 seed=45955、
            # py3.13 seed=47441），于是合成序列不同、beta_perm/k_pred 不可跨版本复现。
            # 注意：TCPD 部分显式用 seed=0，因此不受影响；论文 Table 4 / fig3 只读 TCPD 行。
            # 统一为与其它 runner 一致的确定性方案。
            seed = (ci * 100 + rep) % 100000
            y, true_cps = S.gen_series(n, k, nz, a, s, seed=seed)
            tol = max(5, int(round(0.01 * n)))
            for cfg in CONFIGS:
                for use_floor in [True, False]:
                    cps, info = run_config(y, n, cfg, seed, use_floor)
                    # precision/recall（与 run_synth 相同口径）
                    used = [False] * len(cps)
                    tp = 0
                    for t in true_cps:
                        best = None
                        for i, g in enumerate(cps):
                            if not used[i] and abs(t - g) <= tol:
                                if best is None or abs(t - cps[best]) > abs(t - g):
                                    best = i
                        if best is not None:
                            used[best] = True
                            tp += 1
                    prec = sum(used) / len(cps) if len(cps) else 1.0
                    rec = tp / len(true_cps) if true_cps else 1.0
                    rows.append({"data": f"syn_{nz}_s{s}", "rep": rep, "n": n, "k": k,
                                 "config": f"B{cfg['B']}_a{cfg['alpha']}_L{cfg['L']}",
                                 "floor": use_floor, "k_pred": len(cps),
                                 "k_true": len(true_cps), "precision": prec,
                                 "recall": rec, "beta_perm": info.get("beta_perm_last"),
                                 "beta_final": info.get("beta_last")})
        print(f"syn cell {nz} s{s} done", flush=True)

    # --- TCPD 部分（seed=0，floor on，全部配置）---
    for ds in D.BENCH_DATASETS:
        try:
            y, n = D.load_series(ds)
        except FileNotFoundError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        ann = D.load_annotations(ds)
        for cfg in CONFIGS:
            for use_floor in [True, False]:
                cps, info = run_config(y, n, cfg, 0, use_floor)
                f1, p, r = M.f_measure(ann, cps.tolist(), return_PR=True)
                rows.append({"data": f"tcpd_{ds}", "rep": 0, "n": n, "k": None,
                             "config": f"B{cfg['B']}_a{cfg['alpha']}_L{cfg['L']}",
                             "floor": use_floor, "k_pred": len(cps),
                             "k_true": None, "precision": p, "recall": r, "f1": f1,
                             "beta_perm": info.get("beta_perm_last"),
                             "beta_final": info.get("beta_last")})
        print(f"tcpd {ds} done", flush=True)

    OUT.write_text(json.dumps(rows))
    print(f"saved {len(rows)} rows -> {OUT}")

    # 汇总（合成：P/R；TCPD：F1）
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        key = (r["data"].split("_")[0] if r["data"].startswith("syn") else "tcpd",
               r["config"], r["floor"])
        agg[key].append(r)
    print("\n=== 敏感性汇总 ===")
    for (typ, cfg, fl), v in sorted(agg.items()):
        if typ == "syn":
            print(f"syn  {cfg:16s} floor={fl}: P={np.mean([r['precision'] for r in v]):.3f} "
                  f"R={np.mean([r['recall'] for r in v]):.3f} kerr={np.mean([r['k_pred']-r['k_true'] for r in v]):+8.2f}")
        else:
            print(f"tcpd {cfg:16s} floor={fl}: F1={np.mean([r['f1'] for r in v]):.4f} "
                  f"k={np.mean([r['k_pred'] for r in v]):.2f}")


if __name__ == "__main__":
    main()
