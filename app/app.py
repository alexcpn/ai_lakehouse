"""
FDA FAERS Pharmacovigilance Signal Detection — Streamlit Web UI

A free, no-install tool for querying drug safety signals from FDA
adverse event reports. Hosted on HuggingFace Spaces.

Modes:
  1. Single Drug Query — safety profile for one drug
  2. Drug Class Comparison — side-by-side signal comparison (e.g., VMAT2 inhibitors)
"""

import os

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# Optional AI report generation
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# --- Configuration ---

HF_REPO = os.environ.get("HF_DATASET_REPO", "alexcpn/fda-faers-parquet")
LOCAL_PARQUET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'faers_data', 'parquet'))

FAERS_TABLES = ['DEMO', 'DRUG', 'REAC', 'OUTC', 'RPSR', 'THER', 'INDI']


# --- Page Config ---

st.set_page_config(
    page_title="FAERS Signal Detection",
    page_icon="\u2695",
    layout="wide",
)

# Force white background / black text on all dataframes
st.markdown("""
<style>
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background-color: #ffffff;
    }
    .stDataFrame td, .stDataFrame th {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    .stDataFrame th {
        background-color: #f0f0f0 !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Database Connection ---

@st.cache_resource
def get_connection():
    """Create DuckDB connection with FAERS tables. Cached across sessions."""
    conn = duckdb.connect(database=':memory:')

    if HF_REPO:
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        base = f"hf://datasets/{HF_REPO}/data"
    else:
        base = LOCAL_PARQUET_DIR

    available = []
    for table in FAERS_TABLES:
        if HF_REPO:
            path = f"{base}/{table}.parquet"
        else:
            path = os.path.join(base, f'{table}.parquet')
            if not os.path.exists(path):
                continue
        try:
            conn.execute(f"CREATE VIEW {table.lower()}_raw AS SELECT * FROM read_parquet('{path}')")
            available.append(table)
        except Exception as e:
            st.warning(f"Could not load {table}: {e}")

    if 'DEMO' not in available:
        st.error("DEMO table not found. Please check data source configuration.")
        st.stop()

    # Deduplicated DEMO
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

    for table in available:
        if table != 'DEMO':
            conn.execute(f"CREATE VIEW {table.lower()} AS SELECT * FROM {table.lower()}_raw")

    # Core analysis view
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

    return conn, available


@st.cache_data
def get_dataset_stats(_conn):
    """Get basic dataset statistics. Cached."""
    row = _conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM demo) as unique_cases,
            (SELECT COUNT(*) FROM demo_raw) as total_reports,
            (SELECT COUNT(DISTINCT UPPER(COALESCE(NULLIF(TRIM(prod_ai),''), drugname)))
             FROM drug WHERE role_cod = 'PS') as unique_drugs
    """).fetchone()
    return {
        'unique_cases': row[0],
        'total_reports': row[1],
        'unique_drugs': row[2],
    }


# --- Signal Detection Functions ---

def compute_drug_signals(conn, drug_filter_sql, min_cases=3):
    """Compute PRR/ROR for a specific drug filter.

    drug_filter_sql: SQL WHERE clause fragment for matching drugs,
    e.g., "UPPER(drugname) LIKE '%VALBENAZINE%' OR UPPER(prod_ai) LIKE '%VALBENAZINE%'"
    """
    # Pre-compute scalars to avoid CTE cross-join issues
    N = conn.execute("SELECT COUNT(DISTINCT primaryid) FROM drug_ae").fetchone()[0]
    dt = conn.execute(f"""
        SELECT COUNT(DISTINCT primaryid) FROM drug_ae WHERE {drug_filter_sql}
    """).fetchone()[0]

    if dt == 0:
        return pd.DataFrame()

    query = f"""
        WITH
        ae_for_drug AS (
            SELECT
                UPPER(adverse_event) as adverse_event,
                COUNT(DISTINCT primaryid) as a
            FROM drug_ae
            WHERE {drug_filter_sql}
            GROUP BY 1
            HAVING COUNT(DISTINCT primaryid) >= {min_cases}
        ),
        ae_totals AS (
            SELECT
                UPPER(adverse_event) as adverse_event,
                COUNT(DISTINCT primaryid) as ae_total
            FROM drug_ae
            GROUP BY 1
        )
        SELECT
            af.adverse_event,
            af.a,
            GREATEST({dt} - af.a, 1) as b,
            GREATEST(aet.ae_total - af.a, 1) as c,
            GREATEST({N} - {dt} - aet.ae_total + af.a, 1) as d,
            {dt} as drug_total,
            aet.ae_total,
            {N} as N,

            ROUND((CAST(af.a AS DOUBLE) / {dt}) /
                  (CAST(GREATEST(aet.ae_total - af.a, 1) AS DOUBLE) / GREATEST({N} - {dt}, 1)), 4) as PRR,

            ROUND((CAST(af.a AS DOUBLE) * GREATEST({N} - {dt} - aet.ae_total + af.a, 1)) /
                  (CAST(GREATEST({dt} - af.a, 1) AS DOUBLE) * GREATEST(aet.ae_total - af.a, 1)), 4) as ROR,

            ROUND(POWER(
                af.a * GREATEST({N} - {dt} - aet.ae_total + af.a, 1) -
                GREATEST({dt} - af.a, 1) * GREATEST(aet.ae_total - af.a, 1), 2
            ) * {N} / (
                CAST({dt} AS DOUBLE) * GREATEST({N} - {dt}, 1) *
                aet.ae_total * GREATEST({N} - aet.ae_total, 1)
            ), 4) as chi_squared

        FROM ae_for_drug af
        JOIN ae_totals aet ON af.adverse_event = aet.adverse_event
        WHERE aet.ae_total > af.a
          AND {dt} > af.a
        ORDER BY af.a DESC
    """

    df = conn.execute(query).fetchdf()

    if df.empty:
        return df

    # Post-processing
    df['log_PRR'] = np.log(df['PRR'].clip(lower=1e-10))
    df['log2_PRR'] = np.log2(df['PRR'].clip(lower=1e-10))
    df['se_log_PRR'] = np.sqrt(
        1.0 / df['a'] - 1.0 / (df['a'] + df['b'])
        + 1.0 / df['c'] - 1.0 / (df['c'] + df['d'])
    )
    df['PRR_lower_CI'] = np.exp(df['log_PRR'] - 1.96 * df['se_log_PRR'])
    df['PRR_upper_CI'] = np.exp(df['log_PRR'] + 1.96 * df['se_log_PRR'])
    df['p_value'] = 1 - stats.chi2.cdf(df['chi_squared'], df=1)
    df['neg_log10_p'] = -np.log10(df['p_value'].clip(lower=1e-300))
    df['is_signal'] = (df['PRR'] >= 2.0) & (df['chi_squared'] >= 4.0) & (df['a'] >= min_cases)

    return df


def get_drug_demographics(conn, drug_filter_sql):
    """Get demographics for a drug filter."""
    demo_df = conn.execute(f"""
        SELECT
            CASE WHEN UPPER(sex) = 'F' THEN 'Female'
                 WHEN UPPER(sex) = 'M' THEN 'Male'
                 ELSE 'Unknown' END as sex_label,
            CASE
                WHEN age_years < 18 THEN 'Pediatric (<18)'
                WHEN age_years BETWEEN 18 AND 44 THEN 'Young Adult (18-44)'
                WHEN age_years BETWEEN 45 AND 64 THEN 'Middle Age (45-64)'
                WHEN age_years >= 65 THEN 'Elderly (65+)'
                ELSE 'Unknown'
            END as age_group,
            COUNT(DISTINCT primaryid) as count
        FROM (
            SELECT primaryid, sex,
                CASE
                    WHEN UPPER(age_cod) = 'YR' THEN TRY_CAST(age AS DOUBLE)
                    WHEN UPPER(age_cod) = 'MON' THEN TRY_CAST(age AS DOUBLE) / 12.0
                    WHEN UPPER(age_cod) = 'DEC' THEN TRY_CAST(age AS DOUBLE) * 10.0
                    WHEN UPPER(age_cod) IN ('DY', 'DAY') THEN TRY_CAST(age AS DOUBLE) / 365.25
                    ELSE NULL
                END as age_years
            FROM drug_ae
            WHERE {drug_filter_sql}
        )
        GROUP BY 1, 2
    """).fetchdf()
    return demo_df


def get_drug_outcomes(conn, drug_filter_sql):
    """Get outcome severity breakdown for a drug."""
    try:
        outcomes = conn.execute(f"""
            SELECT
                CASE
                    WHEN UPPER(o.outc_cod) = 'DE' THEN 'Death'
                    WHEN UPPER(o.outc_cod) = 'LT' THEN 'Life-threatening'
                    WHEN UPPER(o.outc_cod) = 'HO' THEN 'Hospitalization'
                    WHEN UPPER(o.outc_cod) = 'DS' THEN 'Disability'
                    WHEN UPPER(o.outc_cod) = 'CA' THEN 'Congenital Anomaly'
                    WHEN UPPER(o.outc_cod) = 'RI' THEN 'Required Intervention'
                    WHEN UPPER(o.outc_cod) = 'OT' THEN 'Other Serious'
                    ELSE 'Other/Unknown'
                END as outcome,
                COUNT(DISTINCT da.primaryid) as count
            FROM drug_ae da
            JOIN outc o ON da.primaryid = o.primaryid
            WHERE {drug_filter_sql}
            GROUP BY 1
            ORDER BY 2 DESC
        """).fetchdf()
        return outcomes
    except Exception:
        return pd.DataFrame()


def build_drug_filter(drug_name):
    """Build SQL WHERE clause for drug name substring matching."""
    safe = drug_name.replace("'", "''").strip().upper()
    return f"(UPPER(drugname) LIKE '%{safe}%' OR UPPER(COALESCE(prod_ai,'')) LIKE '%{safe}%')"


# --- AI Report Generation ---

REPORT_SYSTEM_PROMPT = """You are a pharmacovigilance scientist writing a safety signal report based on FDA FAERS
(FDA Adverse Event Reporting System) disproportionality analysis data.

Write a publication-quality safety signal report that could serve as a starting point for a manuscript section
or regulatory briefing document. Use the following structure:

1. **Background** — Brief description of the drug(s), approved indications, and pharmacological class
2. **Methods** — Mention FAERS database, time period, disproportionality analysis (PRR, ROR), Evans' criteria
   (PRR≥2, χ²≥4, n≥3), and 95% confidence intervals via log-normal approximation
3. **Key Findings** — Discuss the most clinically significant signals, distinguishing between:
   - Known/labeled adverse events (expected from mechanism of action or drug label)
   - Potentially novel signals warranting further investigation
   - Signals likely due to confounding (indication bias, protopathic bias, notoriety bias)
4. **Clinical Context** — How these signals compare to known safety profiles, published literature, or drug labels
5. **Limitations** — FAERS is voluntary reporting (stimulated reporting, Weber effect), cannot establish causation,
   no denominator data (report counts ≠ incidence rates), duplicate reports possible despite deduplication

Be specific with numbers — cite PRR values, confidence intervals, case counts, and p-values from the data provided.
Flag any signals that appear to be indication bias (e.g., antidepressant→depression).
Use formal scientific tone suitable for a clinical pharmacology audience."""


def get_ai_config():
    """Render sidebar AI configuration and return (provider, client) or (None, None)."""
    with st.sidebar:
        st.markdown("### AI Report Generation")
        st.caption("Optional: add your API key to generate publication-quality AI safety reports from the statistical results.")

        provider = st.selectbox(
            "LLM Provider",
            ["None (stats only)", "Anthropic (Claude)", "OpenAI (GPT)"],
            key="ai_provider"
        )

        if provider == "Anthropic (Claude)":
            if not HAS_ANTHROPIC:
                st.warning("anthropic package not installed")
                return None, None
            api_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
            if api_key:
                return "anthropic", anthropic.Anthropic(api_key=api_key)
        elif provider == "OpenAI (GPT)":
            if not HAS_OPENAI:
                st.warning("openai package not installed")
                return None, None
            api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
            model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], key="openai_model")
            if api_key:
                return "openai", openai.OpenAI(api_key=api_key)

    return None, None


def build_report_prompt(drug_name, signals_df, demographics_df, outcomes_df, dataset_stats, report_count):
    """Build the user prompt with all statistical data for the LLM."""
    # Top signals
    sig_df = signals_df[signals_df['is_signal']].nlargest(20, 'PRR')
    nonsig_df = signals_df[~signals_df['is_signal']].nlargest(10, 'a')

    sig_table = ""
    for _, row in sig_df.iterrows():
        sig_table += (
            f"  - {row['adverse_event']}: PRR={row['PRR']:.2f} "
            f"(95% CI: {row['PRR_lower_CI']:.2f}-{row['PRR_upper_CI']:.2f}), "
            f"ROR={row['ROR']:.2f}, χ²={row['chi_squared']:.1f}, "
            f"p={row['p_value']:.2e}, n={int(row['a'])} cases\n"
        )

    nonsig_table = ""
    for _, row in nonsig_df.iterrows():
        nonsig_table += (
            f"  - {row['adverse_event']}: PRR={row['PRR']:.2f} "
            f"(95% CI: {row['PRR_lower_CI']:.2f}-{row['PRR_upper_CI']:.2f}), "
            f"n={int(row['a'])} cases — NOT a signal\n"
        )

    # Demographics summary
    demo_text = ""
    if not demographics_df.empty:
        sex_summary = demographics_df.groupby('sex_label')['count'].sum()
        total = sex_summary.sum()
        demo_text = f"Total reports: {report_count}\n"
        for sex, count in sex_summary.items():
            demo_text += f"  {sex}: {count} ({count*100/total:.1f}%)\n"

    # Outcomes summary
    outcome_text = ""
    if outcomes_df is not None and not outcomes_df.empty:
        for _, row in outcomes_df.iterrows():
            outcome_text += f"  {row['outcome']}: {int(row['count'])} cases\n"

    prompt = f"""Analyze the following FAERS disproportionality analysis results for: **{drug_name}**

## Dataset
- Total unique cases in database: {dataset_stats['unique_cases']:,}
- Total unique drugs: {dataset_stats['unique_drugs']:,}
- Reports for {drug_name}: {report_count:,}

## Demographics
{demo_text}

## Outcome Severity
{outcome_text if outcome_text else "Not available"}

## Signals Detected (Evans' criteria met: PRR≥2, χ²≥4, n≥3)
{len(sig_df)} signals out of {len(signals_df)} adverse events analyzed:
{sig_table if sig_table else "  No signals detected."}

## Most Frequent Non-Signal Adverse Events
{nonsig_table if nonsig_table else "  None."}

Generate a publication-quality safety signal report for {drug_name}."""

    return prompt


def build_comparison_report_prompt(drug_names, comparison_data, report_counts, dataset_stats, class_label):
    """Build prompt for drug class comparison report."""
    prompt = f"""Analyze the following FAERS comparative disproportionality analysis for drug class: **{class_label}**

## Dataset
- Total unique cases in database: {dataset_stats['unique_cases']:,}
- Drugs compared: {', '.join(drug_names)}

## Report Counts
"""
    for drug in drug_names:
        prompt += f"- {drug}: {report_counts.get(drug, 0):,} reports\n"

    # Collect all AEs across drugs that are signals in at least one drug
    all_signal_aes = set()
    for drug, sigs in comparison_data.items():
        if not sigs.empty:
            sig_aes = sigs[sigs['is_signal']]['adverse_event'].tolist()
            all_signal_aes.update(sig_aes)

    prompt += "\n## Comparative Signal Data\n"
    for ae in sorted(all_signal_aes)[:30]:
        prompt += f"\n**{ae}:**\n"
        for drug in drug_names:
            sigs = comparison_data.get(drug, pd.DataFrame())
            if not sigs.empty:
                ae_row = sigs[sigs['adverse_event'] == ae]
                if not ae_row.empty:
                    row = ae_row.iloc[0]
                    is_sig = "SIGNAL" if row['is_signal'] else "no signal"
                    prompt += (
                        f"  - {drug}: PRR={row['PRR']:.2f} "
                        f"(95% CI: {row['PRR_lower_CI']:.2f}-{row['PRR_upper_CI']:.2f}), "
                        f"n={int(row['a'])}, {is_sig}\n"
                    )
                else:
                    prompt += f"  - {drug}: insufficient data\n"
            else:
                prompt += f"  - {drug}: no data\n"

    prompt += f"""
Generate a publication-quality comparative safety report for the {class_label} drug class,
highlighting differential safety signals across the drugs. Identify which adverse events are
class effects (signals across all drugs) vs drug-specific signals."""

    return prompt


def generate_ai_report(provider, client, user_prompt):
    """Call the LLM API and return the report text."""
    try:
        if provider == "anthropic":
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=REPORT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

        elif provider == "openai":
            model = st.session_state.get("openai_model", "gpt-4o")
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content or "No content in response."

    except Exception as e:
        return f"**Error generating report:** {str(e)}"

    return "**Error:** No response from LLM."


# --- UI Components ---

def render_disclaimer():
    st.warning(
        "**Disclaimer:** FAERS is a voluntary reporting system. Report counts do not reflect "
        "true incidence rates. Disproportionality signals (PRR/ROR) indicate statistical "
        "associations, **not causation**. Results require validation by clinical experts "
        "and should not be used for direct clinical decision-making.",
        icon="\u26a0\ufe0f"
    )


def render_volcano_plot(signals_df, title="Drug-AE Signal Detection"):
    """Render a volcano plot for signal detection results."""
    fig, ax = plt.subplots(figsize=(7, 4))

    non_sig = signals_df[~signals_df['is_signal']]
    sig = signals_df[signals_df['is_signal']]

    ax.scatter(non_sig['log2_PRR'], non_sig['neg_log10_p'],
               alpha=0.3, s=20, c='gray', label='Non-signal')
    if not sig.empty:
        ax.scatter(sig['log2_PRR'], sig['neg_log10_p'],
                   alpha=0.7, s=30, c='red', label='Signal (Evans\' criteria)')

        # Label top signals
        for _, row in sig.nlargest(5, 'PRR').iterrows():
            ax.annotate(row['adverse_event'][:20],
                        (row['log2_PRR'], row['neg_log10_p']),
                        fontsize=7, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')

    ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=np.log2(2), color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('log2(PRR)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def render_forest_plot(comparison_data, ae_list, drug_names):
    """Render a forest plot comparing PRR across drugs for given AEs."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(ae_list) * 0.5)))

    cmap = plt.colormaps['Set1']
    colors = [cmap(i / max(len(drug_names) - 1, 1)) for i in range(len(drug_names))]
    y_positions = np.arange(len(ae_list))
    bar_height = 0.8 / len(drug_names)

    for i, drug in enumerate(drug_names):
        drug_data = comparison_data.get(drug, pd.DataFrame())
        prrs = []
        lower_cis = []
        upper_cis = []

        for ae in ae_list:
            row = drug_data[drug_data['adverse_event'] == ae]
            if not row.empty:
                prrs.append(float(row['PRR'].iloc[0]))
                lower_cis.append(float(row['PRR_lower_CI'].iloc[0]))
                upper_cis.append(float(row['PRR_upper_CI'].iloc[0]))
            else:
                prrs.append(0)
                lower_cis.append(0)
                upper_cis.append(0)

        y = y_positions + i * bar_height - 0.4 + bar_height / 2
        xerr_lower = [max(0, p - l) for p, l in zip(prrs, lower_cis)]
        xerr_upper = [u - p for p, u in zip(prrs, upper_cis)]

        ax.errorbar(prrs, y, xerr=[xerr_lower, xerr_upper],
                     fmt='o', color=colors[i], label=drug, markersize=6, capsize=3)

    ax.axvline(x=1, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=2, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='PRR=2 threshold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels([ae[:30] for ae in ae_list], fontsize=8)
    ax.set_xlabel('PRR (95% CI)')
    ax.set_title('Comparative Signal Detection — PRR with 95% CI')
    ax.legend(fontsize=8, loc='lower right')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# --- Main App ---

def main():
    st.title("\u2695 FDA FAERS Signal Detection")
    st.caption("Free pharmacovigilance signal detection from FDA adverse event reports")

    conn, available_tables = get_connection()
    dataset_stats = get_dataset_stats(conn)

    # AI configuration in sidebar
    ai_provider, ai_client = get_ai_config()

    st.markdown(
        f"**Dataset:** {dataset_stats['unique_cases']:,} unique cases | "
        f"{dataset_stats['unique_drugs']:,} unique drugs | "
        f"Tables: {', '.join(available_tables)}"
    )

    render_disclaimer()

    # --- Mode Selection ---
    tab1, tab2 = st.tabs(["Single Drug Query", "Drug Class Comparison"])

    # ==========================================
    # TAB 1: Single Drug Query
    # ==========================================
    with tab1:
        drug_input = st.text_input(
            "Enter drug name (brand or generic):",
            placeholder="e.g., valbenazine, ibuprofen, INGREZZA",
            key="single_drug"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            min_cases = st.slider("Minimum case count", 1, 20, 3, key="single_min")
        with col2:
            pass

        # Run analysis and store in session_state
        if drug_input and st.button("Analyze", key="btn_single"):
            drug_filter = build_drug_filter(drug_input)
            count = conn.execute(f"""
                SELECT COUNT(DISTINCT primaryid) FROM drug_ae WHERE {drug_filter}
            """).fetchone()[0]

            if count == 0:
                st.session_state['single_results'] = None
                st.error(f"No reports found for '{drug_input}'. Try a different spelling or the generic name.")
            else:
                with st.spinner("Computing disproportionality analysis..."):
                    signals = compute_drug_signals(conn, drug_filter, min_cases=min_cases)
                    demo_data = get_drug_demographics(conn, drug_filter)
                    outcomes = get_drug_outcomes(conn, drug_filter)
                st.session_state['single_results'] = {
                    'drug_name': drug_input,
                    'drug_filter': drug_filter,
                    'count': count,
                    'signals': signals,
                    'demo_data': demo_data,
                    'outcomes': outcomes,
                    'min_cases': min_cases,
                }

        # Display results from session_state
        results = st.session_state.get('single_results')
        if results is not None:
            drug_name_display = results['drug_name']
            count = results['count']
            signals = results['signals']
            demo_data = results['demo_data']
            outcomes = results['outcomes']
            min_cases_used = results['min_cases']

            st.success(f"Found **{count:,}** adverse event reports for '{drug_name_display}'")

            # Drug profile
            st.subheader("Drug Profile")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Demographics (Sex)**")
                sex_summary = demo_data.groupby('sex_label')['count'].sum().reset_index()
                st.dataframe(sex_summary, hide_index=True)

            with col_b:
                st.markdown("**Demographics (Age)**")
                age_summary = demo_data.groupby('age_group')['count'].sum().reset_index()
                st.dataframe(age_summary, hide_index=True)

            # Outcomes
            if not outcomes.empty:
                st.subheader("Outcome Severity")
                st.bar_chart(outcomes.set_index('outcome')['count'])

            # Signal detection
            st.subheader("Adverse Event Signals (PRR/ROR)")

            if signals.empty:
                st.info("No adverse events with sufficient case count.")
            else:
                num_signals = int(signals['is_signal'].sum())
                st.markdown(
                    f"**{len(signals)}** adverse events analyzed | "
                    f"**{num_signals}** signals detected (Evans' criteria: PRR\u22652, \u03c7\u00b2\u22654, n\u2265{min_cases_used})"
                )

                # Prepare display data
                display_cols = ['adverse_event', 'a', 'PRR', 'PRR_lower_CI', 'PRR_upper_CI',
                                'ROR', 'chi_squared', 'p_value', 'is_signal']
                display_df = signals[display_cols].copy()
                display_df.columns = ['Adverse Event', 'Cases', 'PRR', 'PRR Lower CI', 'PRR Upper CI',
                                      'ROR', 'Chi-squared', 'P-value', 'Signal']
                display_df = display_df.sort_values(by='PRR', ascending=False)

                # --- PRR Bar Chart (top adverse events) ---
                st.markdown("#### Top Adverse Events by PRR")
                chart_df = display_df.head(20).copy()
                fig, ax = plt.subplots(figsize=(8, max(3, len(chart_df) * 0.3)))
                bar_colors = ['#d32f2f' if s else '#757575' for s in chart_df['Signal']]
                y_pos = range(len(chart_df))
                ax.barh(y_pos, chart_df['PRR'], color=bar_colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(chart_df['Adverse Event'].str[:35], fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel('PRR')
                ax.axvline(x=2, color='blue', linestyle='--', linewidth=0.8, alpha=0.6, label='PRR=2 threshold')
                ax.legend(fontsize=7)
                for i, (prr, n) in enumerate(zip(chart_df['PRR'], chart_df['Cases'])):
                    ax.text(prr + 0.1, i, f'n={int(n)}', va='center', fontsize=7, color='#333333')
                ax.set_title(f'Signal Detection: {drug_name_display}', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # --- Volcano plot ---
                if len(signals) > 10:
                    with st.expander("Volcano Plot", expanded=False):
                        fig = render_volcano_plot(signals, title=f"Signal Detection: {drug_name_display}")
                        st.pyplot(fig)
                        plt.close(fig)

                # --- Signal Table ---
                st.markdown("#### Full Signal Table")
                styled = display_df.style.apply(
                    lambda row: [
                        'background-color: #ffe0e0; color: #000000'
                        if row['Signal'] else
                        'background-color: #ffffff; color: #000000'
                        for _ in row
                    ],
                    axis=1,
                ).format({
                    'PRR': '{:.2f}', 'PRR Lower CI': '{:.2f}', 'PRR Upper CI': '{:.2f}',
                    'ROR': '{:.2f}', 'Chi-squared': '{:.2f}', 'P-value': '{:.2e}',
                })

                st.dataframe(styled, hide_index=True, use_container_width=True)

                # CSV download
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "Download results as CSV",
                    csv,
                    f"faers_signals_{drug_name_display.replace(' ', '_')}.csv",
                    "text/csv"
                )

                # --- AI Report Generation ---
                st.divider()
                if ai_provider and ai_client:
                    st.subheader("AI Safety Report")
                    if st.button("Generate Publication-Quality Report", key="btn_ai_single"):
                        with st.spinner("Generating AI safety report..."):
                            prompt = build_report_prompt(
                                drug_name_display, signals, demo_data,
                                outcomes, dataset_stats, count
                            )
                            report = generate_ai_report(ai_provider, ai_client, prompt)
                            st.session_state['single_ai_report'] = report

                    # Display persisted report
                    if 'single_ai_report' in st.session_state:
                        st.markdown(st.session_state['single_ai_report'])
                        st.download_button(
                            "Download report as text",
                            st.session_state['single_ai_report'],
                            f"faers_ai_report_{drug_name_display.replace(' ', '_')}.md",
                            "text/markdown",
                            key="dl_ai_single"
                        )
                else:
                    st.info(
                        "Add your Anthropic or OpenAI API key in the sidebar to generate "
                        "an AI-powered publication-quality safety report from these results."
                    )

    # ==========================================
    # TAB 2: Drug Class Comparison
    # ==========================================
    with tab2:
        drugs_input = st.text_input(
            "Enter drug names (comma-separated):",
            placeholder="e.g., tetrabenazine, deutetrabenazine, valbenazine",
            key="class_drugs"
        )
        class_label = st.text_input(
            "Class label (optional):",
            placeholder="e.g., VMAT2 Inhibitors",
            key="class_label"
        )
        ae_focus = st.text_input(
            "Focus on specific adverse events (optional, comma-separated):",
            placeholder="e.g., depression, suicidal ideation, suicide attempt",
            key="ae_focus"
        )

        min_cases_class = st.slider("Minimum case count", 1, 20, 3, key="class_min")

        # Run comparison and store in session_state
        if drugs_input and st.button("Compare", key="btn_compare"):
            drug_names = [d.strip() for d in drugs_input.split(',') if d.strip()]
            label = class_label or "Drug Comparison"

            if len(drug_names) < 2:
                st.session_state['compare_results'] = None
                st.error("Please enter at least 2 drugs to compare.")
            else:
                comparison_data = {}
                report_counts = {}

                progress = st.progress(0)
                for i, drug in enumerate(drug_names):
                    drug_filter = build_drug_filter(drug)
                    cnt = conn.execute(f"""
                        SELECT COUNT(DISTINCT primaryid) FROM drug_ae WHERE {drug_filter}
                    """).fetchone()[0]
                    report_counts[drug] = cnt
                    if cnt > 0:
                        comparison_data[drug] = compute_drug_signals(conn, drug_filter, min_cases=min_cases_class)
                    else:
                        comparison_data[drug] = pd.DataFrame()
                    progress.progress((i + 1) / len(drug_names))
                progress.empty()

                # Demographics
                demo_rows = []
                for drug in drug_names:
                    drug_filter = build_drug_filter(drug)
                    demo = get_drug_demographics(conn, drug_filter)
                    sex_counts = demo.groupby('sex_label')['count'].sum()
                    total = sex_counts.sum()
                    female_pct = sex_counts.get('Female', 0) * 100 / total if total else 0
                    male_pct = sex_counts.get('Male', 0) * 100 / total if total else 0
                    demo_rows.append({
                        'Drug': drug,
                        'Total Reports': int(total),
                        'Female %': f"{female_pct:.1f}%",
                        'Male %': f"{male_pct:.1f}%",
                    })

                # Determine AEs
                if ae_focus:
                    focus_aes = [ae.strip().upper() for ae in ae_focus.split(',')]
                else:
                    all_aes = set()
                    for drug, sigs in comparison_data.items():
                        if not sigs.empty:
                            all_aes.update(sigs.nlargest(15, 'a')['adverse_event'].tolist())
                    focus_aes = sorted(all_aes)[:20]

                st.session_state['compare_results'] = {
                    'drug_names': drug_names,
                    'label': label,
                    'comparison_data': comparison_data,
                    'report_counts': report_counts,
                    'demo_rows': demo_rows,
                    'focus_aes': focus_aes,
                }

        # Display comparison results from session_state
        cmp = st.session_state.get('compare_results')
        if cmp is not None:
            drug_names = cmp['drug_names']
            label = cmp['label']
            comparison_data = cmp['comparison_data']
            report_counts = cmp['report_counts']
            demo_rows = cmp['demo_rows']
            focus_aes = cmp['focus_aes']

            st.subheader(f"{label}")

            st.markdown("### Report Counts")
            counts_df = pd.DataFrame([
                {'Drug': drug, 'Reports': cnt}
                for drug, cnt in report_counts.items()
            ])
            st.dataframe(counts_df, hide_index=True)

            st.markdown("### Demographics Comparison")
            st.dataframe(pd.DataFrame(demo_rows), hide_index=True)

            # Comparison table
            st.markdown("### Signal Comparison Table")
            comp_rows = []
            for ae in focus_aes:
                row = {'Adverse Event': ae}
                for drug in drug_names:
                    sigs = comparison_data.get(drug, pd.DataFrame())
                    if not sigs.empty:
                        ae_row = sigs[sigs['adverse_event'] == ae]
                        if not ae_row.empty:
                            prr = float(ae_row['PRR'].iloc[0])
                            ci_lo = float(ae_row['PRR_lower_CI'].iloc[0])
                            ci_hi = float(ae_row['PRR_upper_CI'].iloc[0])
                            n = int(ae_row['a'].iloc[0])
                            is_sig = bool(ae_row['is_signal'].iloc[0])
                            sig_marker = " *" if is_sig else ""
                            row[f'{drug} PRR (95% CI)'] = f"{prr:.2f} ({ci_lo:.2f}-{ci_hi:.2f}){sig_marker}"
                            row[f'{drug} n'] = n
                        else:
                            row[f'{drug} PRR (95% CI)'] = "n/a"
                            row[f'{drug} n'] = 0
                    else:
                        row[f'{drug} PRR (95% CI)'] = "No data"
                        row[f'{drug} n'] = 0
                comp_rows.append(row)

            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            st.caption("\\* = Signal detected (Evans' criteria: PRR\u22652, \u03c7\u00b2\u22654)")

            # Forest plot
            st.markdown("### Forest Plot — PRR with 95% CI")
            if focus_aes and any(not v.empty for v in comparison_data.values()):
                fig = render_forest_plot(comparison_data, focus_aes[:15], drug_names)
                st.pyplot(fig)
                plt.close(fig)

            # Narrative synthesis
            st.markdown("### Summary")
            narrative_parts = []
            for ae in focus_aes[:5]:
                drug_signals = []
                drug_no_signals = []
                for drug in drug_names:
                    sigs = comparison_data.get(drug, pd.DataFrame())
                    if not sigs.empty:
                        ae_row = sigs[sigs['adverse_event'] == ae]
                        if not ae_row.empty and bool(ae_row['is_signal'].iloc[0]):
                            prr = float(ae_row['PRR'].iloc[0])
                            drug_signals.append(f"{drug} (PRR={prr:.2f})")
                        elif not ae_row.empty:
                            prr = float(ae_row['PRR'].iloc[0])
                            drug_no_signals.append(f"{drug} (PRR={prr:.2f})")

                if drug_signals or drug_no_signals:
                    parts = []
                    if drug_signals:
                        parts.append(f"**Signal detected** for {', '.join(drug_signals)}")
                    if drug_no_signals:
                        parts.append(f"no signal for {', '.join(drug_no_signals)}")
                    narrative_parts.append(f"- **{ae}**: {'; '.join(parts)}")

            if narrative_parts:
                st.markdown("\n".join(narrative_parts))

            # CSV download
            csv = comp_df.to_csv(index=False)
            st.download_button(
                "Download comparison as CSV",
                csv,
                f"faers_comparison_{label.replace(' ', '_')}.csv",
                "text/csv"
            )

            # --- AI Report Generation ---
            st.divider()
            if ai_provider and ai_client:
                st.subheader("AI Comparative Safety Report")
                if st.button("Generate Publication-Quality Report", key="btn_ai_compare"):
                    with st.spinner("Generating AI comparative safety report..."):
                        prompt = build_comparison_report_prompt(
                            drug_names, comparison_data, report_counts,
                            dataset_stats, label
                        )
                        report = generate_ai_report(ai_provider, ai_client, prompt)
                        st.session_state['compare_ai_report'] = report

                if 'compare_ai_report' in st.session_state:
                    st.markdown(st.session_state['compare_ai_report'])
                    st.download_button(
                        "Download report as text",
                        st.session_state['compare_ai_report'],
                        f"faers_ai_report_{label.replace(' ', '_')}.md",
                        "text/markdown",
                        key="dl_ai_compare"
                    )
            else:
                st.info(
                    "Add your Anthropic or OpenAI API key in the sidebar to generate "
                    "an AI-powered publication-quality comparative safety report."
                )

    # --- Footer ---
    st.divider()
    st.caption(
        "Data source: [FDA FAERS](https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers). "
        "Methods: Proportional Reporting Ratio (PRR), Reporting Odds Ratio (ROR), Evans' criteria. "
        "AI reports powered by Claude/GPT (bring your own API key). "
        "Built with DuckDB + Streamlit. "
        "Reference: Yokoi et al. 2023, *Psychiatry Clin Neurosci Reports*"
    )


if __name__ == '__main__':
    main()
