"""
FDA FAERS Data Download, Parquet Conversion, and HuggingFace Upload

Downloads quarterly FAERS ASCII data from the FDA, converts dollar-sign
delimited files to Parquet format using DuckDB, and optionally uploads
to a HuggingFace Hub dataset.

Usage:
    # Download a single quarter
    python faers_download.py --year 2024 --quarter 4

    # Download a range of quarters
    python faers_download.py --start 2020Q1 --end 2024Q4

    # Download and upload to HuggingFace Hub
    python faers_download.py --start 2024Q1 --end 2024Q4 --upload --hf-repo user/fda-faers-parquet

Source: https://fis.fda.gov/extensions/FPD-QDE-FAERs/FPD-QDE-FAERs.html
"""

import argparse
import glob
import os
import sys
import zipfile
from datetime import date

import duckdb
import requests

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'faers_data'))
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PARQUET_DIR = os.path.join(DATA_DIR, 'parquet')

FAERS_TABLES = ['DEMO', 'DRUG', 'REAC', 'OUTC', 'RPSR', 'THER', 'INDI']

BASE_URL = "https://fis.fda.gov/content/Exports/"


def parse_quarter_string(s):
    """Parse '2024Q4' or '2024q4' into (year, quarter)."""
    s = s.strip().upper()
    if 'Q' not in s:
        raise ValueError(f"Invalid quarter format: {s}. Expected format: 2024Q4")
    parts = s.split('Q')
    return int(parts[0]), int(parts[1])


def quarter_range(start_year, start_q, end_year, end_q):
    """Generate (year, quarter) tuples from start to end inclusive."""
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        yield y, q
        q += 1
        if q > 4:
            q = 1
            y += 1


def default_quarter():
    """Return the most recent quarter likely available (current minus 2)."""
    today = date.today()
    q = (today.month - 1) // 3 + 1
    y = today.year
    # Go back 2 quarters for availability lag
    q -= 2
    if q <= 0:
        q += 4
        y -= 1
    return y, q


def download_quarter(year, quarter):
    """Download a FAERS quarterly ZIP file. Returns path to ZIP or None on failure."""
    os.makedirs(RAW_DIR, exist_ok=True)

    label = f"{year}Q{quarter}"
    zip_filename = f"faers_ascii_{label}.zip"
    zip_path = os.path.join(RAW_DIR, zip_filename)

    if os.path.exists(zip_path):
        print(f"  Already downloaded: {zip_filename}")
        return zip_path

    # FDA uses inconsistent URL patterns across years
    urls_to_try = [
        f"{BASE_URL}faers_ascii_{year}Q{quarter}.zip",
        f"{BASE_URL}faers_ascii_{year}q{quarter}.zip",
        f"{BASE_URL}faers_ascii_{label}.zip",
    ]

    for url in urls_to_try:
        print(f"  Trying: {url}")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                total = int(resp.headers.get('content-length', 0))
                downloaded = 0
                with open(zip_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r  Downloading {zip_filename}: {pct}%", end='', flush=True)
                print(f"\r  Downloaded: {zip_filename} ({downloaded // (1024*1024)} MB)")
                return zip_path
        except requests.RequestException as e:
            print(f"  Failed: {e}")
            continue

    print(f"  ERROR: Could not download FAERS data for {label}")
    return None


def extract_zip(zip_path, label):
    """Extract ZIP to RAW_DIR/<label>/ subfolder. Returns extraction directory."""
    extract_dir = os.path.join(RAW_DIR, label)
    if os.path.exists(extract_dir):
        print(f"  Already extracted: {label}")
        return extract_dir

    print(f"  Extracting: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    return extract_dir


def find_ascii_files(extract_dir):
    """Locate the 7 FAERS table files within the extracted directory.

    Returns dict mapping table name -> file path.
    Handles case variations and nested directories.
    """
    found = {}
    for table in FAERS_TABLES:
        # Search recursively, case-insensitive
        patterns = [
            os.path.join(extract_dir, '**', f'{table}*.txt'),
            os.path.join(extract_dir, '**', f'{table}*.TXT'),
            os.path.join(extract_dir, '**', f'{table.lower()}*.txt'),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Pick the largest file if multiple matches
                found[table] = max(matches, key=os.path.getsize)
                break

    return found


def convert_to_parquet(ascii_files, quarter_label):
    """Convert dollar-sign delimited ASCII files to Parquet using DuckDB.

    Returns dict mapping table name -> parquet file path.
    """
    os.makedirs(PARQUET_DIR, exist_ok=True)
    conn = duckdb.connect(database=':memory:')
    parquet_files = {}

    for table, filepath in ascii_files.items():
        parquet_path = os.path.join(PARQUET_DIR, f'{table}_{quarter_label}.parquet')
        print(f"  Converting {table}: {os.path.basename(filepath)} -> {os.path.basename(parquet_path)}")

        try:
            # DuckDB reads $-delimited ASCII with auto type detection
            conn.execute(f"""
                COPY (
                    SELECT * FROM read_csv_auto(
                        '{filepath}',
                        delim='$',
                        header=true,
                        ignore_errors=true,
                        sample_size=10000,
                        all_varchar=true
                    )
                ) TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            row_count = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
            ).fetchone()[0]
            print(f"    {row_count:,} rows written")
            parquet_files[table] = parquet_path
        except Exception as e:
            print(f"    ERROR converting {table}: {e}")

    conn.close()
    return parquet_files


def merge_parquet_files(table, file_list, output_path):
    """Merge multiple quarter Parquet files into a single file per table."""
    conn = duckdb.connect(database=':memory:')
    file_glob = "', '".join(file_list)
    print(f"  Merging {table}: {len(file_list)} files -> {os.path.basename(output_path)}")

    try:
        conn.execute(f"""
            COPY (
                SELECT * FROM read_parquet(['{file_glob}'], union_by_name=true)
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        row_count = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
        ).fetchone()[0]
        print(f"    {row_count:,} total rows")
    except Exception as e:
        print(f"    ERROR merging {table}: {e}")

    conn.close()


def deduplicate_demo(parquet_dir):
    """Deduplicate DEMO table: keep latest CASEVERSION per CASEID.

    Overwrites DEMO.parquet in place.
    """
    demo_path = os.path.join(parquet_dir, 'DEMO.parquet')
    if not os.path.exists(demo_path):
        print("  WARNING: DEMO.parquet not found, skipping deduplication")
        return

    print("  Deduplicating DEMO (keeping latest CASEVERSION per CASEID)...")
    conn = duckdb.connect(database=':memory:')

    before = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{demo_path}')").fetchone()[0]

    temp_path = demo_path + '.tmp'
    # Get column names to select all except the _rn helper
    cols = conn.execute(f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{demo_path}'))").fetchall()
    col_list = ', '.join(f'"{c[0]}"' for c in cols)
    conn.execute(f"""
        COPY (
            SELECT {col_list} FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY caseid
                        ORDER BY TRY_CAST(caseversion AS INTEGER) DESC NULLS LAST
                    ) as _rn
                FROM read_parquet('{demo_path}')
            ) WHERE _rn = 1
        ) TO '{temp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    after = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{temp_path}')").fetchone()[0]
    conn.close()

    os.replace(temp_path, demo_path)
    print(f"    Before: {before:,} rows -> After: {after:,} rows ({before - after:,} duplicates removed)")


def upload_to_hf(parquet_dir, hf_repo):
    """Upload merged Parquet files to a HuggingFace Hub dataset."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create the dataset repo if it doesn't exist
    try:
        api.create_repo(hf_repo, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"  Note: {e}")

    for table in FAERS_TABLES:
        parquet_path = os.path.join(parquet_dir, f'{table}.parquet')
        if not os.path.exists(parquet_path):
            print(f"  WARNING: {table}.parquet not found, skipping upload")
            continue

        size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"  Uploading {table}.parquet ({size_mb:.1f} MB) to {hf_repo}...")

        api.upload_file(
            path_or_fileobj=parquet_path,
            path_in_repo=f"data/{table}.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
        )
        print(f"    Uploaded {table}.parquet")

    # Upload a README with dataset card
    readme_content = f"""---
license: cc0-1.0
task_categories:
  - tabular-classification
tags:
  - pharmacovigilance
  - drug-safety
  - fda
  - faers
  - adverse-events
  - signal-detection
pretty_name: FDA FAERS Adverse Event Reports (Parquet)
---

# FDA FAERS Adverse Event Reports

Pre-processed [FDA Adverse Event Reporting System (FAERS)](https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers) data in Parquet format for pharmacovigilance analysis.

## Tables

| Table | Description |
|-------|------------|
| DEMO | Demographics and case admin (deduplicated: latest CASEVERSION per CASEID) |
| DRUG | Drug/medication info (drugname, prod_ai, role_cod, route, dose) |
| REAC | Adverse reactions (MedDRA preferred terms) |
| OUTC | Patient outcomes (death, hospitalization, disability, etc.) |
| RPSR | Report sources |
| THER | Therapy dates |
| INDI | Indications for use |

All tables linked by `primaryid`. Drug role codes: PS=Primary Suspect, SS=Secondary Suspect, C=Concomitant, I=Interacting.

## Usage with DuckDB

```python
import duckdb
conn = duckdb.connect()
conn.install_extension('httpfs')
conn.load_extension('httpfs')

# Query directly from HuggingFace
demo = conn.execute(\"\"\"
    SELECT COUNT(*) as total_cases, COUNT(DISTINCT caseid) as unique_cases
    FROM read_parquet('hf://datasets/{hf_repo}/data/DEMO.parquet')
\"\"\").fetchdf()
```

## Disclaimer

FAERS is a voluntary reporting system. Report counts do not establish causation.
Disproportionality signals (PRR, ROR) indicate statistical associations that require
clinical validation. Not intended for direct clinical decision-making.

## Source

[FDA FAERS Quarterly Data Extract Files](https://fis.fda.gov/extensions/FPD-QDE-FAERs/FPD-QDE-FAERs.html)
"""

    readme_path = os.path.join(parquet_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="dataset",
    )
    print(f"\n  Dataset card uploaded. View at: https://huggingface.co/datasets/{hf_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Download FDA FAERS data, convert to Parquet, optionally upload to HuggingFace"
    )
    parser.add_argument('--year', type=int, help='Single quarter: year (e.g., 2024)')
    parser.add_argument('--quarter', type=int, help='Single quarter: quarter (1-4)')
    parser.add_argument('--start', type=str, help='Range start: 2020Q1')
    parser.add_argument('--end', type=str, help='Range end: 2024Q4')
    parser.add_argument('--upload', action='store_true', help='Upload to HuggingFace Hub')
    parser.add_argument('--hf-repo', type=str, help='HF dataset repo (e.g., user/fda-faers-parquet)')
    args = parser.parse_args()

    # Determine which quarters to download
    quarters = []
    if args.start and args.end:
        sy, sq = parse_quarter_string(args.start)
        ey, eq = parse_quarter_string(args.end)
        quarters = list(quarter_range(sy, sq, ey, eq))
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        y, q = default_quarter()
        print(f"No quarter specified. Defaulting to {y}Q{q}")
        quarters = [(y, q)]

    print(f"\nQuarters to process: {', '.join(f'{y}Q{q}' for y, q in quarters)}")
    print("=" * 60)

    # Step 1: Download and extract each quarter
    all_parquet = {table: [] for table in FAERS_TABLES}

    for year, quarter in quarters:
        label = f"{year}Q{quarter}"
        print(f"\n--- {label} ---")

        zip_path = download_quarter(year, quarter)
        if not zip_path:
            continue

        extract_dir = extract_zip(zip_path, label)
        ascii_files = find_ascii_files(extract_dir)

        if not ascii_files:
            print(f"  WARNING: No ASCII files found in {extract_dir}")
            continue

        print(f"  Found {len(ascii_files)}/{len(FAERS_TABLES)} tables: {', '.join(ascii_files.keys())}")

        parquet_files = convert_to_parquet(ascii_files, label)

        for table, path in parquet_files.items():
            all_parquet[table].append(path)

    # Step 2: Merge quarters into single files per table (if multi-quarter)
    if len(quarters) > 1:
        print("\n" + "=" * 60)
        print("Merging quarters into single Parquet files...")
        for table in FAERS_TABLES:
            files = all_parquet[table]
            if not files:
                continue
            merged_path = os.path.join(PARQUET_DIR, f'{table}.parquet')
            merge_parquet_files(table, files, merged_path)
    else:
        # Single quarter: rename to standard names
        for table in FAERS_TABLES:
            files = all_parquet[table]
            if files:
                standard_path = os.path.join(PARQUET_DIR, f'{table}.parquet')
                if files[0] != standard_path:
                    os.makedirs(os.path.dirname(standard_path), exist_ok=True)
                    os.replace(files[0], standard_path)

    # Step 3: Deduplicate DEMO table
    print("\n" + "=" * 60)
    print("Deduplicating...")
    deduplicate_demo(PARQUET_DIR)

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("Parquet files in", PARQUET_DIR)
    conn = duckdb.connect(database=':memory:')
    for table in FAERS_TABLES:
        path = os.path.join(PARQUET_DIR, f'{table}.parquet')
        if os.path.exists(path):
            rows = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {table:6s}: {rows:>12,} rows  ({size_mb:.1f} MB)")
    conn.close()

    # Step 5: Upload to HuggingFace
    if args.upload:
        if not args.hf_repo:
            print("\nERROR: --hf-repo required when using --upload")
            sys.exit(1)
        print("\n" + "=" * 60)
        print(f"Uploading to HuggingFace: {args.hf_repo}")
        upload_to_hf(PARQUET_DIR, args.hf_repo)

    print("\nDone.")


if __name__ == '__main__':
    main()
