# -*- coding: utf-8 -*-
"""
run_const_sens.py — 地板常数 c 的敏感性扫描 + σ̂ 偏差标定。

两件事（对应审稿 §39 参数敏感性、§58 终审证据审计）：

(A) **σ̂ 标定**：论文 §5.3 / §7 有两处明确写着"MAD 一阶差分 σ̂ 在强正相关下低估有效
    检测尺度，**此解释未经独立验证**"。本脚本直接测量 σ̂ 与真值 σ 的比值，
    把这两个 HYPOTHESIS 升级（或推翻）。

(B) **常数 c 扫描**：UBG-floor 是 β = max(3 ln n, c·ln n·σ̂²)，论文固定 c=2（沿用 BIC
    约定）并在 §5.5 声明"c 的系统研究留作未来工作"。整篇论文的贡献就是这一行，
    若不扫 c，"修好了"就可能是"常数调得好"。本脚本在 c ∈ {1,2,3,4} 上重跑。

网格与 run_uncapped.py 完全一致（同种子公式 seed = ci*100 + rep），
故 c=2 的结果应与 results/raw/uncapped.jsonl 的 n=10³ 半区逐位相同 —— 这是自带的
一致性自检。

输出：results/raw/const_sens.jsonl（逐条追加，可断点续跑）
常量：只用 n=10³ 做 c 扫描（σ=10 处 Q=n//2 单条最慢约 1 s）；σ̂ 标定覆盖 n∈{10³,10⁴}。
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

OUT = ROOT / "results" / "raw" / "const_sens.jsonl"

# 与 run_uncapped.py 保持一致的网格定义（cell 编号 ci 也一致 → 同种子）
NS = [1000, 10000]
KS = [0, 5, 20]
NOISES = ["gauss", "t3", "ar1_09"]
AMPS = [1.0, 3.0]
SIGMAS = [1.0, 10.0]
REPS = 3

CS = [1.0, 2.0, 3.0, 4.0]          # c 扫描
SCALE_NS = [1000]                   # c 扫描只用 n=10³（见模块 docstring）


def cells():
    """与 run_uncapped.py 完全相同的 cell 枚举顺序，保证 seed 一致。"""
    return [(n, k, nz, a, s)
            for n in NS for k in KS for nz in NOISES for a in AMPS for s in SIGMAS]


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["n"], r["k_true_cell"], r["noise"], r["amp"],
                          r["sigma"], r["rep"], r["c"]))
    rows_out = open(OUT, "a")

    def emit(row):
        rows_out.write(json.dumps(row) + "\n")
        rows_out.flush()

    cl = cells()
    for ci, (n, k, nz, a, s) in enumerate(cl, 1):
        for rep in range(REPS):
            seed = ci * 100 + rep
            y, true_cps = S.gen_series(n, k, nz, a, s, seed=seed)
            logn = np.log(n)
            q_inf = max(k + 1, n // 2)

            # (A) σ̂ 标定 —— 对每个序列都记，与 c 无关；只随 c=CS[0] 那次写出，
            #     避免 4 倍冗余。单独存一份 sigmahat 行（c=None）。
            sigmahat = float(L.mad_sigma(y))
            if (n, k, nz, a, s, rep, None) not in done:
                emit({"n": n, "k_true_cell": k, "noise": nz, "amp": a, "sigma": s,
                      "rep": rep, "seed": seed, "c": None,
                      "sigmahat": sigmahat, "sigma_true": s,
                      "ratio": sigmahat / s, "k_true": len(true_cps),
                      "runtime_s": None, "k_pred": None, "pen": None})

            # (B) c 扫描
            if n not in SCALE_NS:
                continue
            for c in CS:
                if (n, k, nz, a, s, rep, c) in done:
                    continue
                pen = max(3.0 * logn, c * logn * sigmahat ** 2)
                t0 = time.perf_counter()
                cps = L.binseg_r(y, pen=pen, Q=q_inf)
                dt = time.perf_counter() - t0
                emit({"n": n, "k_true_cell": k, "noise": nz, "amp": a, "sigma": s,
                      "rep": rep, "seed": seed, "c": c,
                      "sigmahat": sigmahat, "sigma_true": s,
                      "ratio": sigmahat / s, "k_true": len(true_cps),
                      "k_pred": int(len(cps)), "pen": float(pen), "runtime_s": dt})
        print(f"cell {ci}/{len(cl)} done", flush=True)
    rows_out.close()

    # ---------- 汇总 ----------
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]

    print("\n=== (A) σ̂ / σ 比值（MAD 一阶差分 + AR 校正）—— 按 noise × σ ===")
    print(f"{'noise':8s} {'sigma':>6s} {'n':>6s} {'mean σ̂':>9s} {'mean σ̂/σ':>10s} {'n_rep':>6s}")
    for n in NS:
        for nz in NOISES:
            for s in SIGMAS:
                sel = [r for r in rows if r["c"] is None and r["n"] == n
                       and r["noise"] == nz and r["sigma"] == s]
                if not sel:
                    continue
                print(f"{nz:8s} {s:6.1f} {n:6d} "
                      f"{np.mean([r['sigmahat'] for r in sel]):9.3f} "
                      f"{np.mean([r['ratio'] for r in sel]):10.3f} {len(sel):6d}")

    print("\n=== (B) 常数 c 敏感性：k 误差（n=10³, Q=n//2）===")
    print(f"{'noise':8s} {'sigma':>6s} " + " ".join(f"{'c='+str(int(c)):>10s}" for c in CS)
          + f"{'c=动态?':>10s}")
    for nz in NOISES:
        for s in SIGMAS:
            line = f"{nz:8s} {s:6.1f} "
            for c in CS:
                sel = [r for r in rows if r["c"] == c and r["noise"] == nz and r["sigma"] == s]
                if not sel:
                    line += f"{'-':>10s} "
                    continue
                kerr = np.mean([r["k_pred"] - r["k_true"] for r in sel])
                line += f"{kerr:+10.2f} "
            print(line)

    print("\n=== (B') 同上，按 σ 合并（gauss+t3+ar1_09 平均）===")
    for s in SIGMAS:
        line = f"{'ALL':8s} {s:6.1f} "
        for c in CS:
            sel = [r for r in rows if r["c"] == c and r["sigma"] == s]
            if not sel:
                line += f"{'-':>10s} "
                continue
            kerr = np.mean([r["k_pred"] - r["k_true"] for r in sel])
            line += f"{kerr:+10.2f} "
        print(line)

    print("\n=== 自检：c=2 应与 results/raw/uncapped.jsonl 的 n=10³ 半区逐位一致 ===")
    try:
        ref = [json.loads(l) for l in
               (ROOT / "results" / "raw" / "uncapped.jsonl").read_text().splitlines() if l.strip()]
        ref = [r for r in ref if r["algorithm"] == "binseg_qinf_floor" and r["n"] == 1000]
        # 注意：dataset 名形如 syn_1000_0_ar1_09_a1.0_s10.0_2，noise 名自带下划线
        # （ar1_09），必须整体捕获；且键里**必须含 amp**，否则 amp=1/3 两条会互相覆盖。
        import re
        pat = re.compile(r"^syn_(\d+)_(\d+)_(.+?)_a([\d.]+)_s([\d.]+)_(\d+)$")
        refmap = {}
        for r in ref:
            m = pat.match(r["dataset"])
            n_, k_, nz_, a_, s_, rep_ = m.groups()
            refmap[(int(n_), int(k_), nz_, float(a_), float(s_), int(rep_))] = r["k"]
        same = diff = missing = 0
        for r in rows:
            if r["c"] != 2.0:
                continue
            key = (r["n"], r["k_true_cell"], r["noise"], r["amp"], r["sigma"], r["rep"])
            if key not in refmap:
                missing += 1
            elif refmap[key] == r["k_pred"]:
                same += 1
            else:
                diff += 1
        print(f"  一致 {same} / 不一致 {diff} / 参考缺失 {missing}")
    except FileNotFoundError:
        print("  (uncapped.jsonl 不存在，跳过)")


if __name__ == "__main__":
    main()
