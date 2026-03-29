# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository accompanies the article "From Data Lakehouse to AI Data Lakehouse" (`ai-datalakehouse.md`). It demonstrates how AI agents can replace manual analyst workflows by querying structured data in a Lakehouse architecture. Two demos:

1. **NYC Taxi analysis** (`python/duckdb_lakehouse_analysis.py`) — DuckDB + Parquet fare analysis with scipy stats and matplotlib.
2. **FDA FAERS pharmacovigilance** (`python/faers_analysis.py` + `app/app.py`) — drug safety signal detection from FDA adverse event reports using disproportionality analysis (PRR, ROR). Includes a Streamlit web UI for HuggingFace Spaces.

Both follow a **4-round agentic analysis pattern**: schema discovery → frequency/correlation analysis → statistical validation → natural language synthesis.

## Commands

### NYC Taxi Demo
```bash
pip install -r python/requirements.txt
python python/duckdb_lakehouse_analysis.py
```
Requires `yellow_tripdata_2022-01.parquet` in repo root (not committed).

### FAERS — Download Data
```bash
# Single quarter
python python/faers_download.py --year 2024 --quarter 4

# Multi-quarter range
python python/faers_download.py --start 2020Q1 --end 2024Q4

# Download + upload to HuggingFace
python python/faers_download.py --start 2024Q1 --end 2024Q4 --upload --hf-repo user/fda-faers-parquet
```
Downloads from FDA, converts `$`-delimited ASCII → Parquet in `faers_data/parquet/`.

### FAERS — CLI Batch Analysis
```bash
# Local data
python python/faers_analysis.py

# Remote HF data
python python/faers_analysis.py --hf-repo user/fda-faers-parquet
```
Outputs `faers_signal_analysis.png` with volcano plot, signal bars, heatmap.

### FAERS — Streamlit Web UI
```bash
# Local development
pip install -r app/requirements.txt
HF_DATASET_REPO=user/fda-faers-parquet streamlit run app/app.py

# Docker (for HF Spaces deployment)
cd app && docker build -t faers-app . && docker run -p 8501:8501 -e HF_DATASET_REPO=user/fda-faers-parquet faers-app
```

### Trino/K8s Test
```bash
kubectl port-forward svc/trino 8080
python python/test.py
```

## Architecture

### FAERS Signal Detection

The FAERS pipeline uses **disproportionality analysis** — the standard pharmacovigilance method:

- **2x2 contingency table** for each drug-AE pair: a (drug+event), b (drug+no event), c (no drug+event), d (neither)
- **PRR** = (a/(a+b)) / (c/(c+d)) — Proportional Reporting Ratio
- **ROR** = (a*d)/(b*c) — Reporting Odds Ratio
- **Evans' criteria** for signal: PRR ≥ 2, chi² ≥ 4, n ≥ 3
- **95% CI** via log-normal approximation, p-value from chi² (df=1)

Data model: 7 FAERS tables (DEMO, DRUG, REAC, OUTC, RPSR, THER, INDI) linked by `primaryid`. Deduplication keeps latest `caseversion` per `caseid`. Analysis filters to Primary Suspect drugs (`role_cod = 'PS'`).

The Streamlit app (`app/app.py`) reads Parquet from HuggingFace via DuckDB's `hf://` protocol — zero local data needed.

## Domain Expert Guided Build

When a user references `lake_house_skill.md` or asks to build a data analysis application, follow the process in that file. The user is likely a domain expert (doctor, pharmacologist, researcher) — not a developer. Ask them for:

1. **Data source** — URL, file, or description of what to download
2. **Methodology** — paper, formula, or plain-language description of the analysis
3. **Audience** — just them, their team, or the public

Then build the standard stack: ETL script → Parquet → DuckDB → analysis engine → Streamlit UI → optional AI report layer. Use the FAERS tool (`python/faers_download.py`, `python/faers_analysis.py`, `app/app.py`) as the reference implementation.

## Key Data Constraints

- FAERS is voluntary reporting — report counts ≠ incidence rates, signals ≠ causation
- Drug names are non-standardized free text; matching uses substring on `drugname` + `prod_ai`
- FAERS ASCII files use `$` delimiter; `ignore_errors=true` handles malformed rows
- Age field has mixed units (YR, MON, DEC, DY) — normalize via `age_cod` before analysis
- Taxi data: filter outliers `fare_amount > 0 AND < 500`, `trip_distance > 0 AND < 100`
