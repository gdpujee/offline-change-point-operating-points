# -*- coding: utf-8 -*-
"""
check_ubg_floor_scope.py — MAJOR-001 (round 1) scope check for the UBG-floor identity.

Question: does UBG-floor reduce to uncapped BinSeg on *any* z-scored dataset,
or only on the tested TCPD sample?

(i) Counterexample probe: AR(1) phi=-0.8, n=512, seed=0, level shift 0.878 in
    the second half, sample-z-scored; Q=n//2. Compares
    run_uncapped.binseg_floor against uncapped cpd_lib.binseg_r (pen=3 ln n).
    Also re-evaluates the MAD/correlation formula by hand.
(ii) TCPD sample check: mad_sigma^2 for every finite first-channel TCPD series
    (scripts/tcpd_data.load_series, already sample-z-scored); reports the max
    and whether all satisfy 2 ln(n) sigmahat^2 <= 3 ln(n) (floor inactive).

Output: results/raw/ubg_floor_scope.json (overwrite; deterministic, seconds).

NOT_EVIDENCE: self-assert literals mirror prose targets (see MAJOR-004);
this file must never count as number provenance (else traces self-certify).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

import cpd_lib as L  # noqa: E402
import tcpd_data as D  # noqa: E402
from run_uncapped import binseg_floor  # noqa: E402

OUT = ROOT / "results" / "raw" / "ubg_floor_scope.json"


def counterexample():
    rng = np.random.default_rng(0)
    n, phi = 512, -0.8
    eps = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = eps[0] / np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    y[n // 2:] += 0.878
    z = (y - y.mean()) / y.std(ddof=1)
    Q = n // 2

    sig2_impl = float(L.mad_sigma(z) ** 2)

    # Hand re-evaluation of the written MAD/correlation formula.
    d = np.diff(z)
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med))
    var_diff = (mad / np.sqrt(2.0)) ** 2
    c = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    rho = max(0.0, 0.0 if not np.isfinite(c) else c)
    if 0.0 <= rho < 1.0:
        var_diff = var_diff / (1.0 - rho)
    sig2_hand = float(max(var_diff, 1e-12))

    base = L.binseg_r(z, pen=3.0 * np.log(n), Q=Q)
    floor = binseg_floor(z, Q)
    return {
        "n": n, "phi": phi, "seed": 0, "shift": 0.878, "Q": Q,
        "sample_sd": float(z.std(ddof=1)),
        "sigmahat2_impl": sig2_impl,
        "sigmahat2_hand": sig2_hand,
        "floor_active": bool(2.0 * np.log(n) * sig2_impl > 3.0 * np.log(n)),
        "base_cps": [int(c) for c in base],
        "floor_cps": [int(c) for c in floor],
        "identical": bool(np.array_equal(base, floor)),
    }


def tcpd_sample():
    rows = []
    for ds in D.BENCH_DATASETS:
        try:
            y, n_obs = D.load_series(ds)
        except D.MissingDataError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        s2 = float(L.mad_sigma(np.asarray(y, dtype=float)) ** 2)
        n = len(y)
        rows.append({"dataset": ds, "n": n, "sigmahat2": s2,
                     "floor_inactive": bool(2.0 * np.log(n) * s2 <= 3.0 * np.log(n))})
    rows.sort(key=lambda r: -r["sigmahat2"])
    return rows


def main():
    ce = counterexample()
    rows = tcpd_sample()
    out = {
        "counterexample": ce,
        "tcpd": {
            "n_series": len(rows),
            "max_sigmahat2": rows[0]["sigmahat2"] if rows else None,
            "max_dataset": rows[0]["dataset"] if rows else None,
            "all_floor_inactive": bool(all(r["floor_inactive"] for r in rows)),
            "rows": rows,
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    # Round-10 MAJOR-004 provenance mirrors (same in-memory values as above —
    # no second implementation to rot). The prose gate cannot do row
    # arithmetic or nested-object lookup beyond rows_key, so per-series rows
    # and the single counterexample construction are mirrored as JSONL.
    _trows = sorted(rows, key=lambda r: r["dataset"])
    assert len(_trows) == 41, "TCPD series count changed: %d" % len(_trows)
    _tpath = ROOT / "results" / "raw" / "derived_ubg_tcpd.jsonl"
    _tpath.write_text("\n".join(json.dumps(r, sort_keys=True) for r in _trows) + "\n")
    _mx = max(r["sigmahat2"] for r in _trows)
    assert round(_mx, 4) == 1.0316, "max sigmahat2 moved: %.7f" % _mx
    assert round(_mx, 3) == 1.032, "max sigmahat2 bound moved: %.7f" % _mx
    assert _trows and max(_trows, key=lambda r: r["sigmahat2"])["dataset"] == "quality_control_5"
    _cepath = ROOT / "results" / "raw" / "derived_ubg_counterexample.jsonl"
    _cepath.write_text(json.dumps(ce, sort_keys=True) + "\n")
    assert round(ce["sigmahat2_impl"], 2) == 1.99, "counterexample moved: %.7f" % ce["sigmahat2_impl"]
    assert ce["base_cps"] == [253] and ce["floor_cps"] == [] and ce["identical"] is False
    print(f"counterexample: sample_sd={ce['sample_sd']:.4f} "
          f"sigmahat2_impl={ce['sigmahat2_impl']:.6f} hand={ce['sigmahat2_hand']:.6f} "
          f"floor_active={ce['floor_active']} base={ce['base_cps']} "
          f"floor={ce['floor_cps']} identical={ce['identical']}")
    print(f"tcpd: n={out['tcpd']['n_series']} max_sigmahat2={out['tcpd']['max_sigmahat2']:.6f} "
          f"({out['tcpd']['max_dataset']}) all_inactive={out['tcpd']['all_floor_inactive']}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
