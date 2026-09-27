# -*- coding: utf-8 -*-
"""
tcpd_data.py — TCPD 数据加载（与 TCPDBench R utils.R load.dataset 语义一致）。

- 序列取 dataset json 的 series[0].raw（单变量）
- 加载后统一 z-score（R 侧 scale(mat)）
- 标注取 TCPD annotations.json：{dataset: {user_id: [cp, ...]}}（0-based）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "TCPD" / "datasets"
ANN_PATH = Path(__file__).resolve().parent.parent / "data" / "TCPD" / "annotations.json"


class MissingDataError(RuntimeError):
    """Raised when the TCPD corpus -- or the part of it asked for -- is absent.

    Deliberately NOT a subclass of ``FileNotFoundError``. Thirteen scripts under
    ``experiments/`` call ``load_series`` inside ``except FileNotFoundError:
    continue``; a missing corpus was therefore swallowed and the run reported
    success while producing nothing. On a cold unpack of the R57 reproduction
    package, ``run_tcpd_main.py`` printed ``saved 0 rows``, exited 0, and
    overwrote the shipped ``results/raw/tcpd_main.json`` with an empty file
    (R58-1). Escaping those handlers is the entire point of this class.
    """


def _missing(what: str, path) -> MissingDataError:
    return MissingDataError(
        f"TCPD {what} not found: {path}\n"
        "  The corpus ships inside reproduction_package.zip under data/TCPD/.\n"
        "  If you received the code without it, obtain the Turing Change Point\n"
        "  Dataset (Zenodo 10.5281/zenodo.3713242, MIT licence) and unpack it so\n"
        "  that data/TCPD/datasets/<name>/<name>.json and\n"
        "  data/TCPD/annotations.json exist, then re-run. Verify with:\n"
        "      python3 experiments/check_data.py")

# R changepoint 不支持多维，TCPDBench 中多维数据集对这些方法为 SKIP
BENCH_DATASETS = [
    "apple", "bank", "bee_waggle_6", "bitcoin", "brent_spot", "businv",
    "centralia", "children_per_woman", "co2_canada", "construction",
    "debt_ireland", "gdp_argentina", "gdp_croatia", "gdp_iran", "gdp_japan",
    "global_co2", "homeruns", "iceland_tourism", "jfk_passengers",
    "lga_passengers", "measles", "nile", "occupancy", "ozone",
    "quality_control_1", "quality_control_2", "quality_control_3",
    "quality_control_4", "quality_control_5", "rail_lines", "ratner_stock",
    "robocalls", "run_log", "scanline_126007", "scanline_42049", "seatbelts",
    "shanghai_license", "uk_coal_employ", "unemployment_nl", "us_population",
    "usd_isk", "well_log",
]


def load_series(name: str):
    """返回 (z_scored 序列 ndarray, n_obs)。语料缺失时抛 MissingDataError。"""
    p = DATA_ROOT / name / f"{name}.json"
    if not p.exists():
        raise _missing(f"series {name!r}", p)
    d = json.loads(p.read_text())
    vals = [x if x is not None else np.nan for x in d["series"][0]["raw"]]
    y = np.asarray(vals, dtype=float)
    n_obs = int(d["n_obs"])
    assert len(y) == n_obs, f"{name}: len {len(y)} != n_obs {n_obs}"
    # 与 TCPDBench 一致：缺失值由各算法自行处理；这里返回原始 + z-score 两个
    from cpd_lib import zscore

    return zscore(y), n_obs


def load_annotations(name: str) -> dict:
    """{user_id: [cp0based, ...]}。语料缺失时抛 MissingDataError。"""
    if not ANN_PATH.exists():
        raise _missing("annotations.json", ANN_PATH)
    ann = json.loads(ANN_PATH.read_text())
    return {u: list(cps) for u, cps in ann[name].items()}


def load_raw(name: str):
    """未标准化的原始序列（诊断用）。语料缺失时抛 MissingDataError。"""
    p = DATA_ROOT / name / f"{name}.json"
    if not p.exists():
        raise _missing(f"series {name!r}", p)
    d = json.loads(p.read_text())
    vals = [x if x is not None else np.nan for x in d["series"][0]["raw"]]
    return np.asarray(vals, dtype=float)
