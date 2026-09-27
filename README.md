# Offline Change-Point Operating-Point Study

Reproducibility code and generated artifacts accompanying the paper:

**Precision/Recall Operating Points and Penalty Scale Sensitivity in Offline Change-Point Detection: A Mechanism Study of BinSeg and PELT**

Repository: https://github.com/gdpujee/offline-change-point-operating-points

This repository contains selected analysis code, generated result files, processed tables, and figures. It excludes manuscript sources and PDFs, cover letters, submission archives, and review materials.

## Contents

- experiments/: analysis and reproduction scripts
- scripts/: shared implementation and data-loading modules
- results/raw/: stored experiment outputs
- results/processed/: generated tables
- results/figures/: generated figures
- requirements.txt: Python package lower bounds

## Reproduction inputs

The upstream TCPD dataset and TCPDBench source/results are not mirrored here. Obtain them from their maintainers and place them under data/TCPD/ and data/TCPDBench/ in this repository before running experiments that use real series or benchmark reference outputs. See the upstream TCPD record at https://zenodo.org/records/3713242 and TCPDBench record at https://zenodo.org/records/3855165. The project preflight is:

    python3 experiments/check_data.py

For an initial comparison run after installing dependencies and preparing those inputs:

    python3 -m pip install -r requirements.txt
    python3 experiments/validate_defaults.py
    python3 experiments/validate_metrics.py
    python3 experiments/run_tcpd_main.py
    python3 experiments/analyze_tcpd_main.py

The full experiment sequence includes long-running steps and an R reference run. The input data and exact external R setup are not bundled; inspect each script's usage and requirements before launching a full run.

## Integrity

MANIFEST.sha256 records SHA-256 digests for the files in this snapshot. Verify them with:

    shasum -a 256 -c MANIFEST.sha256

## Release metadata

This is a review snapshot. A software license, author citation metadata, and Zenodo DOI have not been assigned in this snapshot; author and rights-holder confirmation is needed before an archival DOI release.
