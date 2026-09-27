#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight for the TCPD corpus -- Appendix B, step 0.

Every script under ``experiments/`` that reads real data goes through
``scripts/tcpd_data.py``. Before R58 the corpus was not shipped inside
``reproduction_package.zip`` (the whole ``data/`` tree was excluded as
"~2 GB"), and thirteen scripts wrapped the loader in
``except FileNotFoundError: continue``. The observable consequence, measured on
a cold unpack of the shipped package: ``run_tcpd_main.py`` printed
``saved 0 rows``, exited 0, and overwrote the shipped
``results/raw/tcpd_main.json`` with an empty file (R58-1). A reproduction chain
that reports success while producing nothing is worse than one that stops.

This script is the preflight that makes the failure loud and actionable *before*
any result file is touched. It checks presence and the upstream md5 manifest.

Usage:  python3 experiments/check_data.py
Exit:   0 = corpus complete and accounted for; 1 = something is missing or
        an undocumented checksum mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "TCPD"
DATASETS = DATA / "datasets"
ANN = DATA / "annotations.json"
CHECKSUMS = DATA / "checksums.json"

# Mirrors scripts/tcpd_data.py:BENCH_DATASETS. Read from the loader rather than
# restated here, so the two cannot drift.
sys.path.insert(0, str(ROOT / "scripts"))

# Upstream declares more than one accepted md5 for this dataset (cross-platform
# float rounding is noted in the upstream README), and the copy in this tree was
# rebuilt by upstream's own get_bee_waggle_6.py from psslds.zip -- its mtime
# differs from the rest of the tree and it matches neither listed digest.
# Documented in README.md and EVIDENCE_LEDGER.md (EVID-014). It is printed every
# run and bounded: any *other* mismatch is a hard failure.
KNOWN_MD5_MISMATCH = {"bee_waggle_6.json"}


def bench_names() -> list[str]:
    try:
        import tcpd_data as D
        return list(D.BENCH_DATASETS)
    except Exception as exc:  # pragma: no cover - loader unavailable
        print("  (cannot import tcpd_data: %s: %s)"
              % (type(exc).__name__, exc), file=sys.stderr)
        return []


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("TCPD corpus preflight (Appendix B, step 0)")
    print("  root: %s" % ROOT)

    names = bench_names()
    if not names:
        print("FATAL: cannot read BENCH_DATASETS from scripts/tcpd_data.py",
              file=sys.stderr)
        return 1
    print("  datasets expected: %d" % len(names))

    missing = [n for n in names if not (DATASETS / n / f"{n}.json").exists()]
    present = len(names) - len(missing)
    print("  datasets present : %d" % present)
    if missing:
        print("  MISSING (%d): %s" % (len(missing), ", ".join(missing)))
        print(
            "\nFATAL: the TCPD corpus is incomplete. It ships inside\n"
            "reproduction_package.zip under data/TCPD/. If you received the\n"
            "code without it, obtain the Turing Change Point Dataset\n"
            "(Zenodo 10.5281/zenodo.3713242, MIT licence) and unpack it so that\n"
            "  data/TCPD/datasets/<name>/<name>.json\n"
            "  data/TCPD/annotations.json\n"
            "exist, then re-run this script.", file=sys.stderr)
        return 1

    for label, p in (("annotations.json", ANN), ("checksums.json", CHECKSUMS)):
        if not p.exists():
            print("  %s: MISSING (%s)" % (label, p))
            print("FATAL: %s is required." % label, file=sys.stderr)
            return 1
        print("  %s: present (%d B)" % (label, p.stat().st_size))

    manifest = json.loads(CHECKSUMS.read_text())
    kind = manifest.get("kind", "?")
    entries = manifest.get("checksums", {})
    print("  manifest: %d entries (%s)" % (len(entries), kind))

    matched, mismatched = 0, []
    for name in sorted(entries):
        p = DATASETS / name[:-5] / name if name.endswith(".json") else None
        if p is None or not p.exists():
            mismatched.append((name, "absent"))
            continue
        got = md5(p)
        want = entries[name]
        want_list = want if isinstance(want, list) else [want]
        if got in want_list:
            matched += 1
        else:
            mismatched.append((name, got))
    print("  md5 match: %d/%d" % (matched, len(entries)))

    undocumented = [nm for nm, _ in mismatched if nm not in KNOWN_MD5_MISMATCH]
    for nm, got in mismatched:
        note = ("documented: upstream lists >1 accepted md5 for this dataset "
                "(cross-platform rounding) and the copy here was rebuilt by "
                "upstream's own build script; see README.md / EVID-014"
                if nm in KNOWN_MD5_MISMATCH else "UNDOCUMENTED")
        print("  md5 mismatch: %s (%s) -- %s" % (nm, got, note))
    # Bounded exemption: the set-aside list is printed above and its size is
    # asserted so it cannot quietly grow into a blind spot.
    if len(mismatched) > len(KNOWN_MD5_MISMATCH) or undocumented:
        print("FATAL: %d undocumented checksum mismatch(es): %s"
              % (len(undocumented), ", ".join(undocumented)), file=sys.stderr)
        return 1

    # The benchmark's own default-parameter outputs: what step 1 compares
    # against. Without them every cell degrades to a status string.
    bench = ROOT / "data" / "TCPDBench" / "abed_results"
    manifest = DATA / "reference_manifest.json"
    needed = ["default_amoc", "default_binseg", "default_pelt",
              "default_segneigh", "default_zero"]
    per_alg: dict[str, int] = {}
    source = ""
    if manifest.exists():
        man = json.loads(manifest.read_text())
        for alg, by_ds in man.get("entries", {}).items():
            per_alg[alg] = sum(len(v) for v in by_ds.values())
        source = "%s (%d runs derived from %s)" % (
            manifest.relative_to(ROOT), man.get("n_source_files", 0),
            man.get("source_glob", "?"))
    elif bench.is_dir():
        for p in bench.glob("*/default_*/*.json"):
            alg = p.parent.name
            per_alg[alg] = per_alg.get(alg, 0) + 1
        source = "%s (%d files)" % (
            bench.relative_to(ROOT), sum(per_alg.values()))
    print("  reference source: %s" % (source or "NONE"))
    if not source:
        print("FATAL: neither %s nor %s is present, so Appendix B step 1 "
              "cannot compare against the benchmark's own outputs."
              % (manifest, bench), file=sys.stderr)
        return 1
    short = [a for a in needed if per_alg.get(a, 0) < 42]
    if short:
        print("FATAL: %s have fewer than 42 reference runs (%s)"
              % (", ".join(short),
                 ", ".join("%s=%d" % (a, per_alg.get(a, 0)) for a in short)),
              file=sys.stderr)
        return 1

    print("OK: corpus complete (%d/%d present, %d/%d md5 verified, "
          "references: %s)"
          % (present, len(names), matched, len(entries), source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
