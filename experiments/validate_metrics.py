# -*- coding: utf-8 -*-
"""validate_metrics.py — 指标实现对拍：我的移植 vs TCPDBench 官方 metrics.py。"""

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import metrics_tcpdbench as mine  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "official_metrics", ROOT / "data" / "TCPDBench" / "analysis" / "scripts" / "metrics.py"
)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def main():
    random.seed(0)
    cases = []
    # 官方 doctest 用例
    cases.append(({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20], 45))
    cases.append(({1: [], 2: [10], 3: [50]}, [10], 45))
    cases.append(({1: [], 2: [10], 3: [50]}, [], 45))
    cases.append(({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45))
    # 随机用例
    for _ in range(300):
        n = random.randint(10, 500)
        k_ann = random.randint(1, 5)
        ann = {
            u: sorted(random.sample(range(1, n - 1), random.randint(0, min(8, n // 5))))
            for u in range(k_ann)
        }
        pred = sorted(random.sample(range(1, n - 1), random.randint(0, min(10, n // 5))))
        cases.append((ann, pred, n))

    bad = 0
    for ann, pred, n in cases:
        f1_a = official.f_measure(ann, pred)
        f1_b = mine.f_measure(ann, pred)
        cov_a = official.covering(ann, pred, n)
        cov_b = mine.covering(ann, pred, n)
        if f1_a != f1_b or cov_a != cov_b:
            bad += 1
            print("MISMATCH", ann, pred, f1_a, f1_b, cov_a, cov_b)
    print(f"checked {len(cases)} cases, mismatches = {bad}")


if __name__ == "__main__":
    main()
