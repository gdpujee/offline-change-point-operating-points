#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the minimal reference manifest that Appendix B step 1 needs.

``validate_defaults.py`` compares this paper's change-point sets against the
benchmark's own default-parameter runs, stored upstream as

    data/TCPDBench/abed_results/<dataset>/default_<algorithm>/<hash>.json

Those files cannot be shipped as-is: each carries a ``command`` and a ``script``
field holding the upstream author's absolute home-directory path, and the
package's leak gate refuses to ship a home path (R58-3). The gate is right, so
rather than widen it this script derives a manifest that keeps only the three
things ``find_default_result`` actually reads --- ``status``, ``parameters`` and
``result.cplocations`` --- and nothing else.

The manifest is deliberately *derived*, not hand-written: run this script to
rebuild it, and the package ships its output.

Usage:  python3 experiments/build_reference_manifest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "TCPDBench" / "abed_results"
OUT = ROOT / "data" / "TCPD" / "reference_manifest.json"

# Fields kept per run. Everything else is dropped: `command` and `script` carry
# home paths, the rest is unused by step 1.
KEEP = ("status", "parameters")


def main() -> int:
    if not SRC.is_dir():
        print("FATAL: %s not found; cannot derive the manifest." % SRC,
              file=sys.stderr)
        return 1

    manifest: dict[str, dict[str, list[dict]]] = {}
    n_files = 0
    for f in sorted(SRC.glob("*/default_*/*.json")):
        alg = f.parent.name            # default_<algorithm>
        ds = f.parent.parent.name      # <dataset>
        rec = json.loads(f.read_text())
        entry = {k: rec.get(k) for k in KEEP}
        entry["cplocations"] = (rec.get("result") or {}).get("cplocations")
        # Sorted by filename, so the "first matching run wins" order that
        # find_default_result depends on is preserved exactly.
        manifest.setdefault(alg, {}).setdefault(ds, []).append(entry)
        n_files += 1

    payload = {
        "generated_by": "experiments/build_reference_manifest.py",
        "source_glob": "data/TCPDBench/abed_results/*/default_*/*.json",
        "n_source_files": n_files,
        "kept_fields": list(KEEP) + ["result.cplocations"],
        "dropped_fields": ["command", "script", "hostname", "error",
                           "dataset_md5", "dataset", "result.runtime"],
        "algorithms": sorted(manifest),
        "entries": manifest,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print("source files: %d" % n_files)
    print("algorithms  : %d" % len(manifest))
    print("datasets    : %d" % len({d for a in manifest.values() for d in a}))
    print("wrote       : %s (%.2f MB)" % (OUT, OUT.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
