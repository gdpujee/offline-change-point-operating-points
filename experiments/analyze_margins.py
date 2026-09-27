# -*- coding: utf-8 -*-
"""
analyze_margins.py — Evaluation across margins M in {1, 3, 5, 10} and Covering metric.

Audits whether the BinSeg vs PELT comparison on TCPD depends on the margin=5 choice.
Computes per-dataset metrics, aggregate means, gaps, win/loss/tie counts, and Wilcoxon p-values.

Outputs:
  - results/raw/margin_sensitivity.json
  - results/processed/margin_sensitivity_table.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import metrics_tcpdbench as M  # noqa: E402
import tcpd_data as D  # noqa: E402

OUT_JSON = ROOT / "results" / "raw" / "margin_sensitivity.json"
OUT_MD = ROOT / "results" / "processed" / "margin_sensitivity_table.md"
MARGINS = [1, 3, 5, 10]


def main():
    tcpd_main = ROOT / "results" / "raw" / "tcpd_main.json"
    assert tcpd_main.exists(), f"missing {tcpd_main}"
    rows = json.loads(tcpd_main.read_text())

    data = defaultdict(dict)
    for r in rows:
        data[r["dataset"]][r["algorithm"]] = r

    datasets = sorted([d for d in D.BENCH_DATASETS if d != "uk_coal_employ"])
    assert len(datasets) == 41, f"expected 41 datasets, got {len(datasets)}"

    per_dataset = {}
    f1_by_margin = {alg: {m: [] for m in MARGINS} for alg in ["binseg", "pelt"]}
    cov_by_alg = {alg: [] for alg in ["binseg", "pelt"]}

    for dname in datasets:
        y, n_obs = D.load_series(dname)
        ann = D.load_annotations(dname)
        d_res = {"dataset": dname, "n_obs": n_obs}

        for alg in ["binseg", "pelt"]:
            cps = data[dname][alg]["cps"]
            d_res[f"{alg}_cps"] = cps
            d_res[f"{alg}_k"] = len(cps)
            for m in MARGINS:
                val = float(M.f_measure(ann, cps, margin=m))
                d_res[f"{alg}_f1_m{m}"] = val
                f1_by_margin[alg][m].append(val)
            cov_val = float(M.covering(ann, cps, n_obs))
            d_res[f"{alg}_covering"] = cov_val
            cov_by_alg[alg].append(cov_val)
        per_dataset[dname] = d_res

    summary = {
        "n_datasets": 41,
        "margins": {},
        "covering": {},
    }

    md_lines = [
        "# Margin and Metric Sensitivity Analysis (TCPD n=41)",
        "",
        "| Metric / Margin | BinSeg Mean | PELT Mean | Gap (BinSeg - PELT) | Win | Loss | Tie | Wilcoxon p |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for m in MARGINS:
        b = np.array(f1_by_margin["binseg"][m])
        p = np.array(f1_by_margin["pelt"][m])
        diff = b - p
        w = int(np.sum(diff > 1e-9))
        l = int(np.sum(diff < -1e-9))
        t = int(len(diff) - w - l)
        p_val = float(st.wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0

        b_mean = float(np.mean(b))
        p_mean = float(np.mean(p))
        gap = float(np.mean(diff))

        summary["margins"][str(m)] = {
            "margin": m,
            "binseg_mean": round(b_mean, 4),
            "pelt_mean": round(p_mean, 4),
            "gap": round(gap, 4),
            "win": w,
            "loss": l,
            "tie": t,
            "wilcoxon_p": round(p_val, 4),
        }
        md_lines.append(
            f"| Margin M={m:2d} | {b_mean:.4f} | {p_mean:.4f} | {gap:+.4f} | {w} | {l} | {t} | {p_val:.4f} |"
        )

    # Covering
    b_c = np.array(cov_by_alg["binseg"])
    p_c = np.array(cov_by_alg["pelt"])
    diff_c = b_c - p_c
    w_c = int(np.sum(diff_c > 1e-9))
    l_c = int(np.sum(diff_c < -1e-9))
    t_c = int(len(diff_c) - w_c - l_c)
    p_val_c = float(st.wilcoxon(diff_c).pvalue) if np.any(np.abs(diff_c) > 1e-12) else 1.0
    b_c_mean = float(np.mean(b_c))
    p_c_mean = float(np.mean(p_c))
    gap_c = float(np.mean(diff_c))

    summary["covering"] = {
        "binseg_mean": round(b_c_mean, 4),
        "pelt_mean": round(p_c_mean, 4),
        "gap": round(gap_c, 4),
        "win": w_c,
        "loss": l_c,
        "tie": t_c,
        "wilcoxon_p": round(p_val_c, 4),
    }
    md_lines.append(
        f"| Covering | {b_c_mean:.4f} | {p_c_mean:.4f} | {gap_c:+.4f} | {w_c} | {l_c} | {t_c} | {p_val_c:.4f} |"
    )

    summary["per_dataset"] = per_dataset

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT_JSON}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"saved {OUT_MD}")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
