"""
AI Data Lakehouse in a Box - FDA FAERS Pharmacovigilance Signal Detection

Demonstrates the agentic analysis pattern from ai-datalakehouse.md:
Query FDA adverse event reports from Parquet files using DuckDB,
perform disproportionality signal detection (PRR, ROR), and generate
publication-quality visualizations.

Works with local Parquet files or remote HuggingFace datasets.

Usage:
    python faers_analysis.py
    python faers_analysis.py --hf-repo user/fda-faers-parquet
"""

import argparse
import os

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PARQUET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'faers_data', 'parquet'))
OUTPUT_IMAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'faers_signal_analysis.png'))

FAERS_TABLES = ['DEMO', 'DRUG', 'REAC', 'OUTC', 'RPSR', 'THER', 'INDI']


def get_connection(hf_repo=None):
    """Create DuckDB connection with FAERS tables registered as views.

    Reads from local Parquet (default) or HuggingFace datasets via hf:// protocol.
    Creates deduplicated DEMO view and joined drug-AE analysis view.
    """
    conn = duckdb.connect(database=':memory:')

    if hf_repo:
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        base = f"hf://datasets/{hf_repo}/data"
    else:
        base = PARQUET_DIR

    # Register raw tables
    available_tables = []
    for table in FAERS_TABLES:
        if hf_repo:
            path = f"{base}/{table}.parquet"
        else:
            path = os.path.join(base, f'{table}.parquet')
            if not os.path.exists(path):
                print(f"  WARNING: {table}.parquet not found, skipping")
                continue
        conn.execute(f"CREATE VIEW {table.lower()}_raw AS SELECT * FROM read_parquet('{path}')")
        available_tables.append(table)

    print(f"  Loaded tables: {', '.join(available_tables)}")

    # Deduplicated DEMO: latest CASEVERSION per CASEID
    demo_cols = conn.execute("SELECT column_name FROM (DESCRIBE SELECT * FROM demo_raw)").fetchall()
    demo_col_list = ', '.join(f'"{c[0]}"' for c in demo_cols)
    conn.execute(f"""
        CREATE VIEW demo AS
        SELECT {demo_col_list} FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY caseid
                    ORDER BY TRY_CAST(caseversion AS INTEGER) DESC NULLS LAST
                ) as _rn
            FROM demo_raw
        ) WHERE _rn = 1
    """)

    # Alias non-demo tables (no dedup needed)
    for table in available_tables:
        if table != 'DEMO':
            conn.execute(f"CREATE VIEW {table.lower()} AS SELECT * FROM {table.lower()}_raw")

    # Core analysis view: deduplicated cases joined with primary suspect drugs and reactions
    conn.execute("""
        CREATE VIEW drug_ae AS
        SELECT
            d.primaryid, d.caseid, d.age, d.age_cod, d.sex, d.wt, d.wt_cod,
            d.event_dt, d.occr_country,
            dr.drugname, dr.prod_ai, dr.role_cod, dr.route,
            r.pt AS adverse_event
        FROM demo d
        JOIN drug dr ON d.primaryid = dr.primaryid
        JOIN reac r ON d.primaryid = r.primaryid
        WHERE dr.role_cod = 'PS'
    """)

    return conn


def round1_schema_and_profiling(conn):
    """Round 1: Schema Discovery and Data Profiling."""

    # Table row counts
    print("\n--- Table Row Counts ---")
    counts = {}
    for table in FAERS_TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table.lower()}").fetchone()[0]
            counts[table] = n
            print(f"  {table:6s}: {n:>12,} rows")
        except Exception:
            counts[table] = 0

    # Deduplication impact
    print("\n--- Deduplication ---")
    dedup = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM demo_raw) as total_reports,
            (SELECT COUNT(*) FROM demo) as unique_cases
    """).fetchdf()
    total = int(dedup['total_reports'].iloc[0])
    unique = int(dedup['unique_cases'].iloc[0])
    dupes = total - unique
    print(f"  Total reports: {total:,}")
    print(f"  Unique cases:  {unique:,}")
    print(f"  Duplicates:    {dupes:,} ({dupes*100/total:.1f}%)")

    # Demographics quality
    print("\n--- Data Quality ---")
    quality = conn.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE age IS NULL OR TRIM(age) = '') as missing_age,
            COUNT(*) FILTER (WHERE sex IS NULL OR TRIM(sex) = '') as missing_sex,
            COUNT(*) FILTER (WHERE wt IS NULL OR TRIM(wt) = '') as missing_weight,
            COUNT(*) FILTER (WHERE occr_country IS NULL OR TRIM(occr_country) = '') as missing_country
        FROM demo
    """).fetchdf()
    total_demo = int(quality['total'].iloc[0])
    missing_age = int(quality['missing_age'].iloc[0])
    missing_sex = int(quality['missing_sex'].iloc[0])
    print(f"  Missing age:     {missing_age:,} ({missing_age*100/total_demo:.1f}%)")
    print(f"  Missing sex:     {missing_sex:,} ({missing_sex*100/total_demo:.1f}%)")
    print(f"  Missing weight:  {int(quality['missing_weight'].iloc[0]):,}")
    print(f"  Missing country: {int(quality['missing_country'].iloc[0]):,}")

    # Sex distribution
    print("\n--- Sex Distribution ---")
    sex_dist = conn.execute("""
        SELECT
            CASE WHEN UPPER(sex) = 'F' THEN 'Female'
                 WHEN UPPER(sex) = 'M' THEN 'Male'
                 ELSE 'Unknown' END as sex_label,
            COUNT(*) as count
        FROM demo
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchdf()
    print(sex_dist.to_string(index=False))

    return {
        'total_cases': total,
        'unique_cases': unique,
        'duplicate_reports': dupes,
        'missing_age_pct': missing_age * 100 / total_demo if total_demo else 0,
        'missing_sex_pct': missing_sex * 100 / total_demo if total_demo else 0,
        'table_counts': counts,
        'sex_distribution': sex_dist,
    }


def round2_drug_ae_frequency(conn):
    """Round 2: Drug-AE Frequency Analysis."""

    # Top 25 drugs
    print("\n--- Top 25 Drugs by Report Count ---")
    top_drugs = conn.execute("""
        SELECT
            UPPER(COALESCE(NULLIF(TRIM(prod_ai), ''), drugname)) as drug_name,
            COUNT(DISTINCT primaryid) as report_count
        FROM drug_ae
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 25
    """).fetchdf()
    print(top_drugs.to_string(index=False))

    # Top 25 adverse events
    print("\n--- Top 25 Adverse Events ---")
    top_aes = conn.execute("""
        SELECT
            UPPER(adverse_event) as adverse_event,
            COUNT(DISTINCT primaryid) as report_count
        FROM drug_ae
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 25
    """).fetchdf()
    print(top_aes.to_string(index=False))

    # Top 50 drug-AE pairs
    print("\n--- Top 50 Drug-AE Pairs ---")
    top_pairs = conn.execute("""
        SELECT
            UPPER(COALESCE(NULLIF(TRIM(prod_ai), ''), drugname)) as drug_name,
            UPPER(adverse_event) as adverse_event,
            COUNT(DISTINCT primaryid) as pair_count
        FROM drug_ae
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT primaryid) >= 3
        ORDER BY 3 DESC
        LIMIT 50
    """).fetchdf()
    print(top_pairs.head(20).to_string(index=False))
    print(f"  ... ({len(top_pairs)} pairs shown)")

    # Age stratification
    print("\n--- Reports by Age Group ---")
    age_strat = conn.execute("""
        SELECT
            CASE
                WHEN age_years < 18 THEN 'Pediatric (<18)'
                WHEN age_years BETWEEN 18 AND 44 THEN 'Young Adult (18-44)'
                WHEN age_years BETWEEN 45 AND 64 THEN 'Middle Age (45-64)'
                WHEN age_years >= 65 THEN 'Elderly (65+)'
                ELSE 'Unknown'
            END as age_group,
            COUNT(DISTINCT primaryid) as report_count
        FROM (
            SELECT primaryid,
                CASE
                    WHEN UPPER(age_cod) = 'YR' THEN TRY_CAST(age AS DOUBLE)
                    WHEN UPPER(age_cod) = 'MON' THEN TRY_CAST(age AS DOUBLE) / 12.0
                    WHEN UPPER(age_cod) = 'DEC' THEN TRY_CAST(age AS DOUBLE) * 10.0
                    WHEN UPPER(age_cod) IN ('DY', 'DAY') THEN TRY_CAST(age AS DOUBLE) / 365.25
                    WHEN UPPER(age_cod) = 'WK' THEN TRY_CAST(age AS DOUBLE) / 52.18
                    WHEN UPPER(age_cod) = 'HR' THEN TRY_CAST(age AS DOUBLE) / 8766.0
                    ELSE NULL
                END as age_years
            FROM drug_ae
        )
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchdf()
    print(age_strat.to_string(index=False))

    # Sex stratification
    print("\n--- Reports by Sex ---")
    sex_strat = conn.execute("""
        SELECT
            CASE WHEN UPPER(sex) = 'F' THEN 'Female'
                 WHEN UPPER(sex) = 'M' THEN 'Male'
                 ELSE 'Unknown' END as sex_label,
            COUNT(DISTINCT primaryid) as report_count
        FROM drug_ae
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchdf()
    print(sex_strat.to_string(index=False))

    return {
        'top_drugs': top_drugs,
        'top_aes': top_aes,
        'top_pairs': top_pairs,
        'age_strat': age_strat,
        'sex_strat': sex_strat,
    }


def compute_signals(conn, min_cases=3):
    """Compute PRR, ROR, chi-squared for all drug-AE pairs.

    Returns DataFrame with signal detection results.
    Shared by both faers_analysis.py and the Streamlit app.
    """
    # Materialize intermediate tables to avoid DuckDB CTE optimization issues
    N = conn.execute("SELECT COUNT(DISTINCT primaryid) FROM drug_ae").fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS _tmp_drug_counts")
    conn.execute("""
        CREATE TEMP TABLE _tmp_drug_counts AS
        SELECT
            UPPER(COALESCE(NULLIF(TRIM(prod_ai), ''), drugname)) as drug_name,
            COUNT(DISTINCT primaryid) as drug_total
        FROM drug_ae
        GROUP BY 1
    """)

    conn.execute("DROP TABLE IF EXISTS _tmp_ae_counts")
    conn.execute("""
        CREATE TEMP TABLE _tmp_ae_counts AS
        SELECT
            UPPER(adverse_event) as adverse_event,
            COUNT(DISTINCT primaryid) as ae_total
        FROM drug_ae
        GROUP BY 1
    """)

    conn.execute("DROP TABLE IF EXISTS _tmp_pair_counts")
    conn.execute(f"""
        CREATE TEMP TABLE _tmp_pair_counts AS
        SELECT
            UPPER(COALESCE(NULLIF(TRIM(prod_ai), ''), drugname)) as drug_name,
            UPPER(adverse_event) as adverse_event,
            COUNT(DISTINCT primaryid) as a
        FROM drug_ae
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT primaryid) >= {min_cases}
    """)

    signals_df = conn.execute(f"""
        SELECT
            p.drug_name,
            p.adverse_event,
            p.a,
            GREATEST(dc.drug_total - p.a, 1) as b,
            GREATEST(ac.ae_total - p.a, 1) as c,
            GREATEST({N} - dc.drug_total - ac.ae_total + p.a, 1) as d,
            dc.drug_total,
            ac.ae_total,
            {N} as N,

            -- PRR = (a/(a+b)) / (c/(c+d))
            ROUND(
                (CAST(p.a AS DOUBLE) / dc.drug_total) /
                (CAST(GREATEST(ac.ae_total - p.a, 1) AS DOUBLE) / GREATEST({N} - dc.drug_total, 1)),
            4) as PRR,

            -- ROR = (a*d) / (b*c)
            ROUND(
                (CAST(p.a AS DOUBLE) * GREATEST({N} - dc.drug_total - ac.ae_total + p.a, 1)) /
                (CAST(GREATEST(dc.drug_total - p.a, 1) AS DOUBLE) * GREATEST(ac.ae_total - p.a, 1)),
            4) as ROR,

            -- Chi-squared
            ROUND(
                POWER(
                    p.a * GREATEST({N} - dc.drug_total - ac.ae_total + p.a, 1) -
                    GREATEST(dc.drug_total - p.a, 1) * GREATEST(ac.ae_total - p.a, 1), 2
                ) * {N} / (
                    CAST(dc.drug_total AS DOUBLE) * GREATEST({N} - dc.drug_total, 1) *
                    ac.ae_total * GREATEST({N} - ac.ae_total, 1)
                ),
            4) as chi_squared

        FROM _tmp_pair_counts p
        JOIN _tmp_drug_counts dc ON p.drug_name = dc.drug_name
        JOIN _tmp_ae_counts ac ON p.adverse_event = ac.adverse_event
        WHERE dc.drug_total > p.a
          AND ac.ae_total > p.a
          AND {N} > dc.drug_total
        ORDER BY PRR DESC
    """).fetchdf()

    # Clean up temp tables
    conn.execute("DROP TABLE IF EXISTS _tmp_drug_counts")
    conn.execute("DROP TABLE IF EXISTS _tmp_ae_counts")
    conn.execute("DROP TABLE IF EXISTS _tmp_pair_counts")

    if signals_df.empty:
        return signals_df

    # Python post-processing: 95% CI and p-values
    signals_df['log_PRR'] = np.log(signals_df['PRR'].clip(lower=1e-10))
    signals_df['log2_PRR'] = np.log2(signals_df['PRR'].clip(lower=1e-10))

    # Standard error of log(PRR) for confidence interval
    signals_df['se_log_PRR'] = np.sqrt(
        1.0 / signals_df['a'] - 1.0 / (signals_df['a'] + signals_df['b'])
        + 1.0 / signals_df['c'] - 1.0 / (signals_df['c'] + signals_df['d'])
    )

    signals_df['PRR_lower_CI'] = np.exp(signals_df['log_PRR'] - 1.96 * signals_df['se_log_PRR'])
    signals_df['PRR_upper_CI'] = np.exp(signals_df['log_PRR'] + 1.96 * signals_df['se_log_PRR'])

    # P-value from chi-squared (1 df)
    signals_df['p_value'] = 1 - stats.chi2.cdf(signals_df['chi_squared'], df=1)
    signals_df['neg_log10_p'] = -np.log10(signals_df['p_value'].clip(lower=1e-300))

    # Evans' criteria: PRR >= 2, chi² >= 4, n >= 3
    signals_df['is_signal'] = (
        (signals_df['PRR'] >= 2.0) &
        (signals_df['chi_squared'] >= 4.0) &
        (signals_df['a'] >= min_cases)
    )

    return signals_df


def round3_signal_detection(conn, r2):
    """Round 3: Disproportionality Signal Detection with Visualization."""

    print("\n--- Computing PRR/ROR for all drug-AE pairs (n >= 3) ---")
    signals_df = compute_signals(conn, min_cases=3)

    if signals_df.empty:
        print("  No drug-AE pairs found with sufficient case counts.")
        return {'signals': signals_df, 'num_signals': 0, 'total_pairs_tested': 0, 'top_signals': signals_df}

    total_pairs = len(signals_df)
    num_signals = int(signals_df['is_signal'].sum())

    print(f"  Total drug-AE pairs tested: {total_pairs:,}")
    print(f"  Signals detected (Evans' criteria): {num_signals:,}")
    print(f"  Signal rate: {num_signals*100/total_pairs:.1f}%")

    # Top signals
    detected = signals_df[signals_df['is_signal']].copy()
    top_signals = detected.nlargest(20, 'PRR')

    print("\n--- Top 20 Signals by PRR ---")
    display_cols = ['drug_name', 'adverse_event', 'a', 'PRR', 'PRR_lower_CI', 'PRR_upper_CI', 'ROR', 'chi_squared', 'p_value']
    print(top_signals[display_cols].to_string(index=False))

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('FDA FAERS Pharmacovigilance Signal Detection', fontsize=14, fontweight='bold')

    # Panel 1: Volcano plot
    ax = axes[0, 0]
    non_sig = signals_df[~signals_df['is_signal']]
    sig = signals_df[signals_df['is_signal']]
    ax.scatter(non_sig['log2_PRR'], non_sig['neg_log10_p'], alpha=0.1, s=5, c='gray', label='Non-signal', rasterized=True)
    if not sig.empty:
        ax.scatter(sig['log2_PRR'], sig['neg_log10_p'], alpha=0.3, s=10, c='red', label='Signal', rasterized=True)
    ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=np.log2(2), color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('log2(PRR)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Volcano Plot: Drug-AE Signal Detection')
    ax.legend(fontsize=8)
    ax.set_xlim(-5, 15)

    # Panel 2: Top 15 signals by PRR
    ax = axes[0, 1]
    top15 = detected.nlargest(15, 'PRR') if not detected.empty else pd.DataFrame()
    if not top15.empty:
        labels = (top15['drug_name'].str[:15] + ' / ' + top15['adverse_event'].str[:15])
        bars = ax.barh(range(len(top15)), top15['PRR'], color='firebrick')
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('PRR')
        ax.set_title('Top 15 Signals by PRR')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'No signals detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Top 15 Signals by PRR')

    # Panel 3: Drug-AE heatmap (top 10 drugs x top 10 AEs among signals)
    ax = axes[1, 0]
    if not detected.empty and len(detected) > 1:
        top_drugs = detected.groupby('drug_name')['a'].sum().nlargest(10).index
        top_aes = detected.groupby('adverse_event')['a'].sum().nlargest(10).index
        heatmap_data = detected[
            detected['drug_name'].isin(top_drugs) & detected['adverse_event'].isin(top_aes)
        ].pivot_table(index='drug_name', columns='adverse_event', values='log2_PRR', fill_value=0)

        if heatmap_data.size > 0:
            im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(heatmap_data.columns)))
            ax.set_xticklabels([c[:12] for c in heatmap_data.columns], rotation=45, ha='right', fontsize=6)
            ax.set_yticks(range(len(heatmap_data.index)))
            ax.set_yticklabels([i[:15] for i in heatmap_data.index], fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8, label='log2(PRR)')
        ax.set_title('Drug-AE Heatmap (log2 PRR)')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Drug-AE Heatmap (log2 PRR)')

    # Panel 4: Reports by sex
    ax = axes[1, 1]
    sex_data = r2['sex_strat']
    colors = {'Female': '#e377c2', 'Male': '#1f77b4', 'Unknown': '#7f7f7f'}
    bar_colors = [colors.get(s, '#7f7f7f') for s in sex_data['sex_label']]
    ax.bar(sex_data['sex_label'], sex_data['report_count'], color=bar_colors)
    ax.set_xlabel('Sex')
    ax.set_ylabel('Report Count')
    ax.set_title('Adverse Event Reports by Sex')
    for i, (_, row) in enumerate(sex_data.iterrows()):
        ax.text(i, row['report_count'], f"{row['report_count']:,}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {OUTPUT_IMAGE}")
    plt.close()

    return {
        'signals': signals_df,
        'num_signals': num_signals,
        'total_pairs_tested': total_pairs,
        'top_signals': top_signals,
    }


def round4_synthesis(r1, r2, r3):
    """Round 4: Natural Language Synthesis of Findings."""

    top = r3['top_signals']
    num_signals = r3['num_signals']
    total_pairs = r3['total_pairs_tested']

    signal_rate = f"{num_signals*100/total_pairs:.1f}%" if total_pairs > 0 else "N/A"

    summary = f"""
FINDINGS SUMMARY
================
Dataset: {r1['unique_cases']:,} unique FAERS cases ({r1['total_cases']:,} total reports, {r1['duplicate_reports']:,} follow-up duplicates removed)
Data Quality: {r1['missing_age_pct']:.1f}% missing age, {r1['missing_sex_pct']:.1f}% missing sex

1. REPORTING LANDSCAPE
   Top drug by reports: {r2['top_drugs'].iloc[0]['drug_name']} ({int(r2['top_drugs'].iloc[0]['report_count']):,} reports)
   Top adverse event: {r2['top_aes'].iloc[0]['adverse_event']} ({int(r2['top_aes'].iloc[0]['report_count']):,} reports)
   Age profile: {r2['age_strat'].iloc[0]['age_group']} is the most reported group ({int(r2['age_strat'].iloc[0]['report_count']):,} reports)

2. SIGNAL DETECTION
   {total_pairs:,} drug-AE pairs tested (minimum 3 cases each)
   {num_signals:,} signals detected (Evans' criteria: PRR>=2, chi²>=4, n>=3)
   Signal rate: {signal_rate} of tested pairs
"""

    if not top.empty:
        summary += "\n3. TOP 5 SIGNALS BY PRR\n"
        for i, (_, row) in enumerate(top.head(5).iterrows(), 1):
            summary += (
                f"   {i}. {row['drug_name'][:25]:25s} + {row['adverse_event'][:25]:25s}\n"
                f"      PRR={row['PRR']:.1f} (95% CI: {row['PRR_lower_CI']:.1f}-{row['PRR_upper_CI']:.1f}), "
                f"ROR={row['ROR']:.1f}, n={int(row['a'])}, p={row['p_value']:.2e}\n"
            )

    summary += f"""
IMPORTANT LIMITATIONS
- FAERS is voluntary reporting. Signal counts reflect reporting patterns, not incidence rates.
- Disproportionality signals (PRR/ROR) indicate statistical associations, NOT causation.
- Results should be validated by clinical experts before any prescribing decisions.

Visualization saved to: {OUTPUT_IMAGE}
"""
    return summary


def main():
    parser = argparse.ArgumentParser(description="FAERS Pharmacovigilance Signal Detection")
    parser.add_argument('--hf-repo', type=str, help='HuggingFace dataset repo (e.g., user/fda-faers-parquet)')
    args = parser.parse_args()

    source = f"HuggingFace: {args.hf_repo}" if args.hf_repo else f"Local: {PARQUET_DIR}"

    print("=" * 60)
    print("AI Data Lakehouse in a Box — FDA FAERS Pharmacovigilance")
    print(f"Data: {source}")
    print("=" * 60)

    conn = get_connection(hf_repo=args.hf_repo)

    print("\n" + "=" * 60)
    print("ROUND 1: Schema Discovery and Data Profiling")
    print("=" * 60)
    r1 = round1_schema_and_profiling(conn)

    print("\n" + "=" * 60)
    print("ROUND 2: Drug-AE Frequency Analysis")
    print("=" * 60)
    r2 = round2_drug_ae_frequency(conn)

    print("\n" + "=" * 60)
    print("ROUND 3: Signal Detection (PRR/ROR)")
    print("=" * 60)
    r3 = round3_signal_detection(conn, r2)

    print("\n" + "=" * 60)
    print("ROUND 4: Synthesis")
    print("=" * 60)
    summary = round4_synthesis(r1, r2, r3)
    print(summary)

    conn.close()


if __name__ == '__main__':
    main()
