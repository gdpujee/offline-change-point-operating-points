# -*- coding: utf-8 -*-
"""run_r_reference.py — 真的把 R 跑一次。

背景
----
一直到 Round 47，稿件里都写着 ``R is not installed on our machine``，复现证据链
只有一层：我们的 Python 实现 vs TCPDBench *发布的* raw 输出（`abed_results/`）。
外部评审指出这是缺口 —— 官方流水线是 R + `changepoint` 2.2.2/2.2.4，而我们从未
真正执行过它。本脚本补上这一层。

做什么
------
对每个 TCPD 数据集 × 4 个算法，用**官方脚本原样的命令行**跑一次 R：

    R --no-echo --no-restore --no-save --slave \
      --file=data/TCPDBench/execs/R/cpdbench_changepoint.R \
      --args -i <dataset>.json -p MBIC -f mean -t Normal -m <Method> [-Q default]

命令行取自官方结果文件里记录的 ``command`` 字段（`abed_results/<ds>/default_binseg/*.json`），
不是我们编的。然后对每条记录做三方比对：

    R 实跑输出  vs  官方发布的 raw 输出  vs  我们的 Python 实现

输出
----
- ``results/raw/r_reference.jsonl``  —— 每次 R 调用的**原始 stdout**（逐字保存）,
  外加解析后的 cplocations 与三项两两比对结果。这是 raw 证据，不可手工改。
  **唯一一处声明过的改写**：R 的 stdout 会回显仓库的绝对路径,故写入前把
  ``<REPO_ROOT>/`` 前缀替换掉(``_redact_root()``)。替换是确定性且无损的
  —— 仓库根是已知的;除了这一处前缀,stdout 其余部分逐字保留。
- ``results/raw/r_reference_summary.json`` —— 汇总计数。

环境
----
- R 4.6.1（`/opt/homebrew/bin/R`），`changepoint` **2.2.2**（CRAN Archive，源码编译）。
- 依赖装在个人 R 库（默认 ``~/R_libs``，可用 ``TCPD_R_LIBS`` 覆盖），通过 ``R_LIBS``
  环境变量指过去。**源码里不写死任何 home 路径** —— 投稿包有绝对路径泄漏扫描
  （``scripts/make_submission_package.py``），写死了会把复现包构建卡住。
- changepoint 2.2.2 的 C 源码是 K&R 旧式函数定义，clang 17 默认的 ``-std=gnu23``
  会拒绝它；``~/.R/Makevars`` 里写 ``CFLAGS=-std=gnu17`` 才能编过。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cpd_lib as L  # noqa: E402
import tcpd_data as D  # noqa: E402

R_BIN = os.environ.get("TCPD_R_BIN", "/opt/homebrew/bin/R")
R_LIBS = os.environ.get("TCPD_R_LIBS", str(Path.home() / "R_libs"))
R_SCRIPT = ROOT / "data" / "TCPDBench" / "execs" / "R" / "cpdbench_changepoint.R"
# 注意：是 data/TCPD/datasets（42 个 json 齐全），不是 data/alan-turing-institute-TCPD-42a33d2/datasets
# （后者只有 32 个，其余 10 个是 get_*.py 下载脚本、没有落盘数据）。
TCPD_DIR = ROOT / "data" / "TCPD" / "datasets"
ABED = ROOT / "data" / "TCPDBench" / "abed_results"

# Output paths are env-overridable so the same pipeline can be re-run against a
# different `changepoint` version without editing the runner. Default keeps the
# 2.2.2 evidence file exactly where it is.
OUT_JSONL = Path(os.environ.get("TCPD_OUT_JSONL",
                                str(ROOT / "results" / "raw" / "r_reference.jsonl")))
OUT_SUMMARY = Path(os.environ.get("TCPD_OUT_SUMMARY",
                                  str(ROOT / "results" / "raw" / "r_reference_summary.json")))

# (our_name, R method name, extra args, penalty, expected Q)
METHODS = [
    ("amoc", "AMOC", [], "MBIC", None),
    ("binseg", "BinSeg", ["-Q", "default"], "MBIC", 5),
    ("pelt", "PELT", [], "MBIC", None),
    ("segneigh", "SegNeigh", ["-Q", "default"], "BIC", 5),
]


def official_reference(ds: str, alg: str):
    """官方发布的 raw 输出（abed_results 里 default_<alg> 那次运行的 cplocations）。"""
    d = ABED / ds / f"default_{alg}"
    if not d.is_dir():
        return None, "no_dir"
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("status") != "SUCCESS":
            return None, r.get("status", "?")
        p = r.get("parameters", {})
        if alg == "amoc" and p.get("penalty") == "MBIC" and p.get("method") == "AMOC":
            return r.get("result", {}).get("cplocations"), "ok"
        if alg == "binseg" and p.get("penalty") == "MBIC" and p.get("method") == "BinSeg" and p.get("Q") == 5:
            return r.get("result", {}).get("cplocations"), "ok"
        if alg == "pelt" and p.get("penalty") == "MBIC" and p.get("method") == "PELT":
            return r.get("result", {}).get("cplocations"), "ok"
        if alg == "segneigh" and p.get("penalty") == "BIC" and p.get("method") == "SegNeigh" and p.get("Q") == 5:
            return r.get("result", {}).get("cplocations"), "ok"
    return None, "no_match"


def _redact_root(text: str) -> str:
    """把仓库根的绝对路径替换成 ``<REPO_ROOT>/``。

    R 的 stdout 会回显我们传给它的绝对路径（脚本路径、数据集路径），这些路径
    带着作者本机的 home 目录。投稿包的泄漏扫描（`make_submission_package.py`）
    会把它判为泄漏并**中止构建**，所以这里在写入前做一次确定的、无损的前缀
    替换（仓库根已知）。除此之外 stdout 逐字保留。

    注意：本文件自身也在被扫描的范围内，所以**注释里不要写出 home 路径的字面量**
    —— 写了同样会触发扫描器（它会命中自己以外的任何文件）。
    """
    return text.replace(str(ROOT) + "/", "<REPO_ROOT>/")


def strip_phantom(locs, alg: str, n_obs: int):
    """去掉 R 导出的尾部哨兵，口径与 validate_defaults.py 完全一致。

    不这么做会造出假的不一致：R 与官方 raw 输出都带哨兵（binseg/pelt/segneigh 尾部
    n-1；amoc 无变点时导出 [n-1] 表示空集），而我们的 Python 实现不带。两侧必须用
    同一把尺子量，所以哨兵剥离施加在 R 输出与官方输出**双方**上。
    """
    if locs is None:
        return None, False
    s = sorted(int(x) for x in locs)
    phantom = False
    if s and s[-1] == n_obs - 1 and alg in ("binseg", "pelt", "segneigh"):
        s = s[:-1]
        phantom = True
    if alg == "amoc" and s == [n_obs - 1]:
        s = []
        phantom = True
    return s, phantom


def my_run(alg: str, y: np.ndarray):
    return {
        "amoc": L.default_amoc,
        "binseg": L.default_binseg,
        "pelt": L.default_pelt,
        "segneigh": L.default_segneigh,
    }[alg](y)


def run_r(dataset_json: Path, r_method: str, extra: list[str], penalty: str = "MBIC"):
    """跑一次官方 R 脚本。返回 (命令, stdout 全文, stderr 末 5 行, 解析后的 dict 或 None)。
    stderr 只用于控制台诊断，不进 JSONL（记录 schema 冻结，避免已提交证据字节漂移；写盘前整行 redact 已覆盖既有字段）。"""
    argv = [
        R_BIN, "--no-echo", "--no-restore", "--no-save", "--slave",
        "--file=" + str(R_SCRIPT),
        "--args",
        "-i", str(dataset_json),
        "-p", penalty,
        "-f", "mean",
        "-t", "Normal",
        "-m", r_method,
    ] + extra
    env = dict(os.environ)
    env["R_LIBS"] = R_LIBS
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=900)
    stdout = proc.stdout
    stderr_tail = proc.stderr.strip().splitlines()[-5:]
    parsed = None
    s = stdout.strip()
    if s.startswith("{"):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
    return " ".join(argv), stdout, stderr_tail, parsed


def main():
    records = []
    summary = {
        alg: {
            "runs": 0,
            "r_success": 0,
            "r_skip_multidim": 0,
            "r_fail": 0,
            "has_official_ref": 0,
            "r_vs_official_exact": 0,
            "r_vs_official_diff": 0,
            "r_vs_official_uncomparable": 0,
            "r_vs_python_exact": 0,
            "r_vs_python_diff": 0,
            "r_vs_python_uncomparable": 0,
        }
        for alg, _, _, _, _ in METHODS
    }

    missing = [ds for ds in D.BENCH_DATASETS
               if not (TCPD_DIR / ds / f"{ds}.json").exists()]
    if missing:
        # Round-8 MINOR-007: an absent corpus used to be skipped silently
        # (0-byte JSONL + rc 0 + step PASS). The corpus gate is step 0
        # (experiments/check_data.py); this runner's contract is the full
        # roster, so a missing file fails loudly here before anything runs
        # or is written (same exists() expression the loop below uses).
        print("missing dataset files: %d (showing up to 10): %s"
              % (len(missing), missing[:10]))
        print("run step 0 (experiments/check_data.py) first; "
              "refusing to write partial outputs")
        return 1

    for ds in D.BENCH_DATASETS:
        djson = TCPD_DIR / ds / f"{ds}.json"
        if not djson.exists():
            continue
        y = None
        n_obs = None
        try:
            y, n_obs = D.load_series(ds)
        except (FileNotFoundError, ValueError, D.MissingDataError):
            y, n_obs = None, None
        finite = y is not None and bool(np.all(np.isfinite(y)))

        for alg, r_method, extra, penalty, exp_q in METHODS:
            cmd, stdout, stderr_tail, parsed = run_r(djson, r_method, extra, penalty)
            r_status = parsed.get("status") if parsed else "UNPARSEABLE"
            if r_status == "UNPARSEABLE":
                print("UNPARSEABLE R output: %s %s :: %s"
                      % (ds, alg, " / ".join(stderr_tail) or "(empty stderr)"))
            r_locs = (parsed.get("result", {}) or {}).get("cplocations") if parsed else None
            r_runtime = (parsed.get("result", {}) or {}).get("runtime") if parsed else None
            r_params = parsed.get("parameters") if parsed else None

            off_locs, off_state = official_reference(ds, alg)

            py_locs = None
            py_state = "no_finite_data" if not finite else "ok"
            if finite:
                try:
                    py_locs = list(map(int, my_run(alg, y)))
                except Exception as exc:  # pragma: no cover - defensive
                    py_state = f"error: {type(exc).__name__}"

            def cmp(a, b):
                if a is None or b is None:
                    return "uncomparable"
                return "exact" if [int(x) for x in a] == [int(x) for x in b] else "diff"

            r_off_raw = cmp(r_locs, off_locs)
            # 哨兵剥离后的科学比对（三方同一口径）
            nn = int(n_obs) if n_obs is not None else None
            r_corr, r_phantom = strip_phantom(r_locs, alg, nn)
            off_corr, off_phantom = strip_phantom(off_locs, alg, nn)
            r_off = cmp(r_corr, off_corr)
            r_py = cmp(r_corr, py_locs)
            py_off = cmp(py_locs, off_corr)

            records.append({
                "dataset": ds,
                "algorithm": alg,
                "r_method": r_method,
                "r_command": cmd,
                "r_status": r_status,
                "r_parameters": r_params,
                "r_cplocations": r_locs,
                "r_runtime_s": r_runtime,
                "official_cplocations": off_locs,
                "official_state": off_state,
                "python_cplocations": py_locs,
                "python_state": py_state,
                "r_cplocations_corrected": r_corr,
                "official_cplocations_corrected": off_corr,
                "phantom_stripped_r": r_phantom,
                "phantom_stripped_official": off_phantom,
                "cmp_r_vs_official_raw": r_off_raw,
                "cmp_r_vs_official": r_off,
                "cmp_r_vs_python": r_py,
                "cmp_python_vs_official": py_off,
                "raw_r_stdout": _redact_root(stdout),
            })

            s = summary[alg]
            s["runs"] += 1
            if r_status == "SUCCESS":
                s["r_success"] += 1
            elif r_status == "SKIP":
                s["r_skip_multidim"] += 1
            else:
                s["r_fail"] += 1
            if off_locs is not None:
                s["has_official_ref"] += 1
            for key, val in (("r_vs_official", r_off), ("r_vs_python", r_py)):
                s[f"{key}_{val}"] = s.get(f"{key}_{val}", 0) + 1

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for rec in records:
            # Redact the WHOLE serialised record, not just stdout. stdout was the
            # only leak when this runner was written, but `r_command` also echoes
            # the repo root; redacting one field at a time is how a second leak
            # survives. Redacting the final line makes it impossible to forget one.
            fh.write(_redact_root(json.dumps(rec, ensure_ascii=False, sort_keys=True)) + "\n")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("records written:", len(records), "->", OUT_JSONL)
    for alg, _, _, _, _ in METHODS:
        print(f"  {alg:9s}", json.dumps(summary[alg], sort_keys=True))
    bad = sum(1 for rec in records if rec["r_status"] == "UNPARSEABLE")
    if bad:
        # Round-7 MAJOR-003: uninterpretable R output used to exit 0 (168/168
        # UNPARSEABLE on a broken R install still reported success). Parsed
        # FAIL/SKIP statuses stay exit-0 (e.g. the known uk_coal_employ FAIL);
        # only UNPARSEABLE — R-side breakage by construction — fails loudly.
        print("UNPARSEABLE R outputs: %d of %d (see lines above)" % (bad, len(records)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
