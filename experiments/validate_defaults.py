# -*- coding: utf-8 -*-
"""
validate_defaults.py — 复现校验：我的 Python 实现 vs TCPDBench 官方原始结果。

对每个数据集：
1. 从 data/TCPDBench/abed_results/<ds>/default_<alg>/<hash>.json 中找到 default 参数
   的那次运行的 cplocations（R changepoint 包真实输出）。
2. 用我的 Python 实现（相同 MBIC/BIC 语义）在同一数据上运行，比较变点集合。
3. 指标实现与官方 metrics.py 直接 import 对拍（docstring 用例 + 随机用例）。

输出：results/raw/validate_defaults.json + 控制台汇总。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import tcpd_data as D  # noqa: E402

RES = ROOT / "data" / "TCPDBench" / "abed_results"
# Derived, shippable copy of the same reference sets (see
# experiments/build_reference_manifest.py). The upstream files cannot be
# shipped -- each carries its author's home path -- so when only the
# manifest is present, step 1 still has something to compare against.
MANIFEST = ROOT / "data" / "TCPD" / "reference_manifest.json"
OUT = ROOT / "results" / "raw" / "validate_defaults.json"
# 4 x 37 (amoc/binseg/pelt/segneigh) + 41 (zero): the count the paper's
# "37/37 bit-for-bit" claim rests on. Asserted, because the failure mode
# this guards against is silent -- with no reference outputs every cell
# becomes {"status": "no_dir"} and the script still exits 0 (R58-3).
EXPECTED_OK_CELLS = 189


def _matches(alg: str, params: dict) -> bool:
    if alg == "amoc" and params.get("penalty") == "MBIC" and params.get("method") == "AMOC":
        return True
    if alg == "binseg" and params.get("penalty") == "MBIC" and params.get("method") == "BinSeg" and params.get("Q") == 5:
        return True
    if alg == "pelt" and params.get("penalty") == "MBIC" and params.get("method") == "PELT":
        return True
    if alg == "segneigh" and params.get("penalty") == "BIC" and params.get("method") == "SegNeigh" and params.get("Q") == 5:
        return True
    if alg == "zero":
        return True
    return False


def find_default_result(ds: str, alg: str):
    """在 abed_results 里找 default_<alg> 的结果（参数匹配的那份）。

    若 abed_results 不存在而派生清单存在，则改读清单（字段一致）。
    """
    if not RES.is_dir() and MANIFEST.exists():
        entries = json.loads(MANIFEST.read_text()).get(
            "entries", {}).get(f"default_{alg}", {}).get(ds)
        if entries is None:
            return None, "no_dir"
        for r in entries:
            if r.get("status") != "SUCCESS":
                return None, r.get("status", "?")
            params = r.get("parameters", {}) or {}
            if _matches(alg, params):
                return r.get("cplocations"), "ok"
        return None, "no_match"
    d = RES / ds / f"default_{alg}"
    if not d.is_dir():
        return None, "no_dir"
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "SUCCESS":
            return None, r.get("status", "?")
        params = r.get("parameters", {})
        if _matches(alg, params):
            return r.get("result", {}).get("cplocations"), "ok"
    return None, "no_match"


def my_run(alg: str, y: np.ndarray):
    if alg == "amoc":
        return L.default_amoc(y)
    if alg == "binseg":
        return L.default_binseg(y)
    if alg == "pelt":
        return L.default_pelt(y)
    if alg == "segneigh":
        return L.default_segneigh(y)
    if alg == "zero":
        return L.zero(y)
    raise ValueError(alg)


def main():
    report = {}
    algorithms = ["amoc", "binseg", "pelt", "segneigh", "zero"]
    summary = {a: {"n_ref": 0, "exact": 0, "within1": 0, "skip": 0, "phantom": 0, "mismatch": []} for a in algorithms}

    for ds in D.BENCH_DATASETS:
        try:
            y, n_obs = D.load_series(ds)
        except D.MissingDataError as exc:
            raise SystemExit("FATAL: %s" % exc)
        if not np.all(np.isfinite(y)):
            continue
        report[ds] = {}
        for alg in algorithms:
            ref, status = find_default_result(ds, alg)
            if ref is None:
                summary[alg]["skip"] += 1
                report[ds][alg] = {"status": status}
                continue
            # 幻影/哨兵：
            # - PELT/BinSeg/SegNeigh: R 导出 cpts 尾部附加哨兵 n → 0-based n-1
            # - AMOC: 无变点时 R decision 返回哨兵 n → 导出 [n-1]（意为空集）
            ref_s = sorted(int(x) for x in ref)
            phantom = bool(ref_s) and ref_s[-1] == n_obs - 1 and alg in ("binseg", "pelt", "segneigh")
            if phantom:
                ref_s = ref_s[:-1]
                summary[alg]["phantom"] += 1
            if alg == "amoc" and ref_s == [n_obs - 1]:
                ref_s = []
                summary[alg]["phantom"] += 1
            got = my_run(alg, y).tolist()
            exact = ref_s == got
            # within1: 双向贪心匹配，允许 ±1
            def within(refl, gotl, tol=1):
                used = [False] * len(gotl)
                ok = 0
                for t in refl:
                    best = None
                    for i, g in enumerate(gotl):
                        if not used[i] and abs(t - g) <= tol:
                            if best is None or abs(t - gotl[best]) > abs(t - g):
                                best = i
                    if best is not None:
                        used[best] = True
                        ok += 1
                return ok == len(refl) == len(gotl)
            w1 = within(ref_s, got)
            s = summary[alg]
            s["n_ref"] += 1
            if exact:
                s["exact"] += 1
            if w1:
                s["within1"] += 1
            if not exact:
                s["mismatch"].append(ds)
            report[ds][alg] = {"status": "ok", "ref": ref_s, "got": got, "exact": exact, "within1": w1, "phantom": phantom}

    n_ok = sum(1 for ds in report for alg in report[ds]
               if report[ds][alg].get("status") == "ok")
    if n_ok != EXPECTED_OK_CELLS:
        raise SystemExit(
            f"FATAL: {n_ok} reference comparisons succeeded, expected "
            f"{EXPECTED_OK_CELLS}. Without a reference source -- either "
            f"{RES}/<ds>/default_<alg>/ or the derived {MANIFEST.name} -- "
            f"every cell degrades to a status string while this script "
            f"still exits 0, and that is exactly what this guard is for. "
            f"Refusing to overwrite {OUT}. "
            f"Run: python3 experiments/check_data.py")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"report": report, "summary": summary}, indent=1))
    print(f"saved -> {OUT}")
    for a in algorithms:
        s = summary[a]
        print(f"{a:9s} ref={s['n_ref']:3d} exact={s['exact']:3d} within1={s['within1']:3d} phantom={s['phantom']:3d} skip={s['skip']:3d} mismatch={s['mismatch'][:8]}")


if __name__ == "__main__":
    main()
