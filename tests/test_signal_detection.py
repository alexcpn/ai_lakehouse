"""
Validation test suite for FAERS pharmacovigilance signal detection.

Tests the core disproportionality analysis math (PRR, ROR, chi-squared,
Evans' criteria) against hand-calculated values and known drug-AE signals
from published literature.

Run: pytest tests/test_signal_detection.py -v
"""

import math

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Import the function under test
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from faers_analysis import compute_signals


# ---------------------------------------------------------------------------
# Helper: build a minimal in-memory FAERS database from a 2x2 table spec
# ---------------------------------------------------------------------------

def build_synthetic_db(drug_ae_pairs):
    """Create a DuckDB connection with synthetic FAERS data from 2x2 specs.

    Args:
        drug_ae_pairs: list of dicts, each with:
            - drug: str
            - ae: str
            - a: int  (drug + event)
            - b: int  (drug + no event)
            - c: int  (no drug + event)
            - d: int  (no drug + no event)

    Returns a DuckDB connection with the drug_ae view expected by compute_signals.
    """
    conn = duckdb.connect(database=':memory:')
    rows = []
    pid = 1

    for pair in drug_ae_pairs:
        drug = pair['drug']
        ae = pair['ae']
        a, b, c, d = pair['a'], pair['b'], pair['c'], pair['d']

        # Cell A: reports with this drug AND this AE
        for _ in range(a):
            rows.append((pid, drug, ae))
            pid += 1

        # Cell B: reports with this drug but a DIFFERENT AE
        for _ in range(b):
            rows.append((pid, drug, f'_OTHER_AE_{pid}'))
            pid += 1

        # Cell C: reports with a DIFFERENT drug but this AE
        for _ in range(c):
            rows.append((pid, f'_OTHER_DRUG_{pid}', ae))
            pid += 1

        # Cell D: reports with a DIFFERENT drug and a DIFFERENT AE
        for _ in range(d):
            rows.append((pid, f'_OTHER_DRUG_{pid}', f'_OTHER_AE_{pid}'))
            pid += 1

    df = pd.DataFrame(rows, columns=['primaryid', 'drugname', 'adverse_event'])
    df['prod_ai'] = ''
    df['role_cod'] = 'PS'
    df['caseid'] = df['primaryid'].astype(str)
    df['age'] = '50'
    df['age_cod'] = 'YR'
    df['sex'] = 'M'
    df['wt'] = '70'
    df['wt_cod'] = 'KG'
    df['event_dt'] = '20240101'
    df['occr_country'] = 'US'
    df['route'] = 'ORAL'

    conn.execute("CREATE TABLE drug_ae AS SELECT * FROM df")

    return conn


def hand_calculate_prr(a, b, c, d):
    """PRR = (a/(a+b)) / (c/(c+d))"""
    return (a / (a + b)) / (c / (c + d))


def hand_calculate_ror(a, b, c, d):
    """ROR = (a*d) / (b*c)"""
    return (a * d) / (b * c)


def hand_calculate_chi2(a, b, c, d):
    """Chi-squared (no Yates correction) = N*(ad - bc)^2 / (R1*R2*C1*C2)
    where R1=a+b, R2=c+d, C1=a+c, C2=b+d, N=a+b+c+d
    """
    N = a + b + c + d
    return (N * (a * d - b * c) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d))


def hand_calculate_se_log_prr(a, b, c, d):
    """Standard error of log(PRR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))"""
    return math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))


# ---------------------------------------------------------------------------
# Test 1: Pure math verification with a simple 2x2 table
# ---------------------------------------------------------------------------

class TestPRRMathBasic:
    """Verify PRR/ROR/chi2 formulas against hand calculations."""

    # Classic textbook example: a=20, b=80, c=100, d=800
    # Drug has 20% AE rate vs 11.1% background => PRR ~1.8
    PAIR = {'drug': 'DRUG_X', 'ae': 'HEADACHE', 'a': 20, 'b': 80, 'c': 100, 'd': 800}

    @pytest.fixture
    def signals(self):
        conn = build_synthetic_db([self.PAIR])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == 'DRUG_X') & (result['adverse_event'] == 'HEADACHE')
        ].iloc[0]
        return row

    def test_prr(self, signals):
        expected = hand_calculate_prr(20, 80, 100, 800)
        assert abs(signals['PRR'] - expected) < 0.01, \
            f"PRR: got {signals['PRR']}, expected {expected:.4f}"

    def test_ror(self, signals):
        expected = hand_calculate_ror(20, 80, 100, 800)
        assert abs(signals['ROR'] - expected) < 0.01, \
            f"ROR: got {signals['ROR']}, expected {expected:.4f}"

    def test_chi_squared(self, signals):
        expected = hand_calculate_chi2(20, 80, 100, 800)
        assert abs(signals['chi_squared'] - expected) < 0.5, \
            f"Chi2: got {signals['chi_squared']}, expected {expected:.4f}"

    def test_p_value(self, signals):
        chi2 = hand_calculate_chi2(20, 80, 100, 800)
        expected_p = 1 - stats.chi2.cdf(chi2, df=1)
        assert abs(signals['p_value'] - expected_p) < 0.01, \
            f"p-value: got {signals['p_value']}, expected {expected_p:.6f}"

    def test_confidence_interval_contains_prr(self, signals):
        assert signals['PRR_lower_CI'] < signals['PRR'] < signals['PRR_upper_CI']

    def test_confidence_interval_width(self, signals):
        """CI should be reasonable — not absurdly wide or collapsed to a point."""
        se = hand_calculate_se_log_prr(20, 80, 100, 800)
        expected_lower = math.exp(math.log(signals['PRR']) - 1.96 * se)
        expected_upper = math.exp(math.log(signals['PRR']) + 1.96 * se)
        assert abs(signals['PRR_lower_CI'] - expected_lower) < 0.05
        assert abs(signals['PRR_upper_CI'] - expected_upper) < 0.05


# ---------------------------------------------------------------------------
# Test 2: Evans' criteria classification
# ---------------------------------------------------------------------------

class TestEvansCriteria:
    """Verify Evans' signal criteria: PRR >= 2, chi² >= 4, n >= 3."""

    def _make_signal(self, a, b, c, d, drug='D', ae='AE'):
        conn = build_synthetic_db([{'drug': drug, 'ae': ae, 'a': a, 'b': b, 'c': c, 'd': d}])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        match = result[(result['drug_name'] == drug) & (result['adverse_event'] == ae)]
        return match

    def test_strong_signal_detected(self):
        """a=50, b=50, c=50, d=850 => PRR=8.5, should be a signal."""
        df = self._make_signal(50, 50, 50, 850)
        assert len(df) == 1
        assert df.iloc[0]['is_signal'] == True
        assert df.iloc[0]['PRR'] >= 2.0

    def test_weak_association_not_signal(self):
        """a=10, b=90, c=200, d=700 => PRR ~0.44, not a signal."""
        df = self._make_signal(10, 90, 200, 700)
        if len(df) > 0:
            assert df.iloc[0]['is_signal'] == False

    def test_too_few_cases_not_signal(self):
        """Only 2 cases — below min_cases=3 threshold, should be excluded."""
        df = self._make_signal(2, 8, 10, 980)
        assert len(df) == 0  # filtered out by min_cases

    def test_high_prr_but_low_chi2_not_signal(self):
        """a=3, b=3, c=300, d=694 => PRR is high but chi2 may be low with low N.
        This tests that PRR alone doesn't make a signal."""
        df = self._make_signal(3, 3, 300, 694)
        if len(df) > 0 and df.iloc[0]['chi_squared'] < 4.0:
            assert df.iloc[0]['is_signal'] == False


# ---------------------------------------------------------------------------
# Test 3: Known drug-AE signals from published literature
# ---------------------------------------------------------------------------

class TestKnownSignals:
    """Validate that known drug-AE associations produce strong signals.

    These use synthetic 2x2 tables with proportions derived from published
    pharmacovigilance literature. The exact counts are synthetic but the
    *ratios* approximate real-world reporting patterns.

    References:
    - Evans SJW et al. (2001) "Use of proportional reporting ratios (PRRs)
      for signal generation from spontaneous adverse drug reaction reports"
    - Rothman KJ et al. (2004) "Reporting of disproportionality analyses"
    - van Puijenbroek EP et al. (2002) "A comparison of measures of
      disproportionality for signal detection in spontaneous reporting
      systems for adverse drug reactions"
    """

    KNOWN_SIGNALS = [
        {
            # Rofecoxib (Vioxx) / Myocardial Infarction — withdrawn 2004
            # Real FAERS data shows very strong disproportionality
            'drug': 'ROFECOXIB', 'ae': 'MYOCARDIAL INFARCTION',
            'a': 150, 'b': 850, 'c': 200, 'd': 8800,
            'expected_prr_min': 2.0,
            'description': 'Rofecoxib/MI — well-established cardiotoxicity signal',
        },
        {
            # Thalidomide / Birth Defects (Phocomelia)
            # Extremely strong historical signal
            'drug': 'THALIDOMIDE', 'ae': 'PHOCOMELIA',
            'a': 80, 'b': 120, 'c': 5, 'd': 9795,
            'expected_prr_min': 10.0,
            'description': 'Thalidomide/phocomelia — strongest known drug-AE signal',
        },
        {
            # Isotretinoin (Accutane) / Birth Defects
            # Well-known teratogenicity signal
            'drug': 'ISOTRETINOIN', 'ae': 'FOETAL MALFORMATION',
            'a': 60, 'b': 340, 'c': 20, 'd': 9580,
            'expected_prr_min': 5.0,
            'description': 'Isotretinoin/birth defects — pregnancy category X',
        },
        {
            # Fluoroquinolones / Tendon Rupture
            # FDA black box warning added 2008
            'drug': 'LEVOFLOXACIN', 'ae': 'TENDON RUPTURE',
            'a': 40, 'b': 460, 'c': 15, 'd': 9485,
            'expected_prr_min': 3.0,
            'description': 'Levofloxacin/tendon rupture — black box warning signal',
        },
        {
            # Statins / Rhabdomyolysis (especially cerivastatin, withdrawn 2001)
            'drug': 'CERIVASTATIN', 'ae': 'RHABDOMYOLYSIS',
            'a': 90, 'b': 110, 'c': 30, 'd': 9770,
            'expected_prr_min': 5.0,
            'description': 'Cerivastatin/rhabdomyolysis — withdrawn from market',
        },
    ]

    @pytest.fixture(params=KNOWN_SIGNALS, ids=[s['description'] for s in KNOWN_SIGNALS])
    def known_signal(self, request):
        spec = request.param
        conn = build_synthetic_db([spec])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == spec['drug']) &
            (result['adverse_event'] == spec['ae'])
        ]
        return row.iloc[0] if len(row) > 0 else None, spec

    def test_signal_detected(self, known_signal):
        row, spec = known_signal
        assert row is not None, f"No result for {spec['description']}"
        assert row['is_signal'] == True, \
            f"{spec['description']}: PRR={row['PRR']}, chi2={row['chi_squared']}, n={row['a']}"

    def test_prr_above_expected_minimum(self, known_signal):
        row, spec = known_signal
        assert row is not None
        assert row['PRR'] >= spec['expected_prr_min'], \
            f"{spec['description']}: PRR={row['PRR']}, expected >= {spec['expected_prr_min']}"

    def test_statistically_significant(self, known_signal):
        row, spec = known_signal
        assert row is not None
        assert row['p_value'] < 0.05, \
            f"{spec['description']}: p={row['p_value']}, expected < 0.05"

    def test_ci_lower_bound_above_1(self, known_signal):
        """For true signals, the lower 95% CI should be > 1 (significant)."""
        row, spec = known_signal
        assert row is not None
        assert row['PRR_lower_CI'] > 1.0, \
            f"{spec['description']}: PRR lower CI={row['PRR_lower_CI']}, expected > 1.0"


# ---------------------------------------------------------------------------
# Test 4: Known NON-signals (negative controls)
# ---------------------------------------------------------------------------

class TestNegativeControls:
    """Pairs with no known association should NOT produce signals."""

    NON_SIGNALS = [
        {
            # Paracetamol/headache — paracetamol TREATS headaches; low disproportionality
            'drug': 'PARACETAMOL', 'ae': 'HEADACHE',
            'a': 50, 'b': 950, 'c': 500, 'd': 8500,
            'description': 'Paracetamol/headache — treats, not causes',
        },
        {
            # Baseline noise — equal rates everywhere
            'drug': 'PLACEBO_DRUG', 'ae': 'COMMON_AE',
            'a': 100, 'b': 900, 'c': 900, 'd': 8100,
            'description': 'Equal proportions — no disproportionality',
        },
    ]

    @pytest.fixture(params=NON_SIGNALS, ids=[s['description'] for s in NON_SIGNALS])
    def non_signal(self, request):
        spec = request.param
        conn = build_synthetic_db([spec])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == spec['drug']) &
            (result['adverse_event'] == spec['ae'])
        ]
        return row.iloc[0] if len(row) > 0 else None, spec

    def test_not_a_signal(self, non_signal):
        row, spec = non_signal
        if row is not None:
            assert row['is_signal'] == False, \
                f"{spec['description']}: incorrectly flagged as signal (PRR={row['PRR']})"

    def test_prr_near_or_below_baseline(self, non_signal):
        row, spec = non_signal
        if row is not None:
            assert row['PRR'] < 2.0, \
                f"{spec['description']}: PRR={row['PRR']}, expected < 2.0"


# ---------------------------------------------------------------------------
# Test 5: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_exact_evans_boundary(self):
        """PRR exactly 2.0, chi2 exactly ~4, n=3 — right at the boundary."""
        # Construct so PRR ≈ 2.0: (a/(a+b)) / (c/(c+d)) = 2
        # If c/(c+d) = 0.1, then a/(a+b) = 0.2
        # a=3, b=12 => 3/15 = 0.2; c=100, d=900 => 100/1000 = 0.1
        conn = build_synthetic_db([{
            'drug': 'BOUNDARY', 'ae': 'EDGE_AE',
            'a': 3, 'b': 12, 'c': 100, 'd': 900,
        }])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == 'BOUNDARY') & (result['adverse_event'] == 'EDGE_AE')
        ]
        assert len(row) > 0, "Boundary case should not be filtered out"

    def test_min_cases_threshold(self):
        """Pairs with exactly min_cases should be included."""
        conn = build_synthetic_db([{
            'drug': 'RARE_DRUG', 'ae': 'RARE_AE',
            'a': 3, 'b': 7, 'c': 10, 'd': 980,
        }])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == 'RARE_DRUG') & (result['adverse_event'] == 'RARE_AE')
        ]
        assert len(row) == 1, "Pair with exactly min_cases=3 should be included"

    def test_below_min_cases_excluded(self):
        """Pairs below min_cases should be excluded."""
        conn = build_synthetic_db([{
            'drug': 'VERY_RARE', 'ae': 'VERY_RARE_AE',
            'a': 2, 'b': 8, 'c': 10, 'd': 980,
        }])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == 'VERY_RARE') & (result['adverse_event'] == 'VERY_RARE_AE')
        ]
        assert len(row) == 0, "Pair below min_cases should be excluded"

    def test_large_n_stability(self):
        """With large counts, PRR should still be numerically stable."""
        conn = build_synthetic_db([{
            'drug': 'BIG_DRUG', 'ae': 'BIG_AE',
            'a': 5000, 'b': 45000, 'c': 10000, 'd': 940000,
        }])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == 'BIG_DRUG') & (result['adverse_event'] == 'BIG_AE')
        ]
        assert len(row) == 1
        expected_prr = hand_calculate_prr(5000, 45000, 10000, 940000)
        assert abs(row.iloc[0]['PRR'] - expected_prr) < 0.01
        assert np.isfinite(row.iloc[0]['PRR'])
        assert np.isfinite(row.iloc[0]['p_value'])

    def test_empty_result(self):
        """No pairs above min_cases => empty DataFrame, no crash."""
        conn = build_synthetic_db([{
            'drug': 'LONELY', 'ae': 'LONELY_AE',
            'a': 1, 'b': 9, 'c': 5, 'd': 985,
        }])
        result = compute_signals(conn, min_cases=5)
        conn.close()
        assert result.empty or len(result[
            (result['drug_name'] == 'LONELY') & (result['adverse_event'] == 'LONELY_AE')
        ]) == 0


# ---------------------------------------------------------------------------
# Test 6: PRR / ROR relationship invariant
# ---------------------------------------------------------------------------

class TestMathInvariants:
    """Mathematical relationships that must always hold."""

    CASES = [
        {'drug': 'D1', 'ae': 'A1', 'a': 30, 'b': 70, 'c': 50, 'd': 850},
        {'drug': 'D2', 'ae': 'A2', 'a': 100, 'b': 200, 'c': 80, 'd': 620},
        {'drug': 'D3', 'ae': 'A3', 'a': 5, 'b': 95, 'c': 50, 'd': 850},
    ]

    @pytest.fixture(params=CASES, ids=['case1', 'case2', 'case3'])
    def signal_row(self, request):
        spec = request.param
        conn = build_synthetic_db([spec])
        result = compute_signals(conn, min_cases=3)
        conn.close()
        row = result[
            (result['drug_name'] == spec['drug']) & (result['adverse_event'] == spec['ae'])
        ]
        return row.iloc[0] if len(row) > 0 else None, spec

    def test_prr_positive(self, signal_row):
        row, _ = signal_row
        assert row is not None
        assert row['PRR'] > 0

    def test_ror_positive(self, signal_row):
        row, _ = signal_row
        assert row is not None
        assert row['ROR'] > 0

    def test_chi2_non_negative(self, signal_row):
        row, _ = signal_row
        assert row is not None
        assert row['chi_squared'] >= 0

    def test_p_value_in_range(self, signal_row):
        row, _ = signal_row
        assert row is not None
        assert 0 <= row['p_value'] <= 1

    def test_ci_ordered(self, signal_row):
        """Lower CI < PRR < Upper CI."""
        row, _ = signal_row
        assert row is not None
        assert row['PRR_lower_CI'] <= row['PRR'] <= row['PRR_upper_CI']

    def test_ror_ge_prr_when_prr_gt_1(self, signal_row):
        """When PRR > 1, ROR >= PRR (a known mathematical property)."""
        row, spec = signal_row
        if row is not None and row['PRR'] > 1:
            assert row['ROR'] >= row['PRR'] - 0.01, \
                f"ROR ({row['ROR']}) should be >= PRR ({row['PRR']}) when PRR > 1"

    def test_ror_le_prr_when_prr_lt_1(self, signal_row):
        """When PRR < 1, ROR <= PRR (a known mathematical property)."""
        row, spec = signal_row
        if row is not None and row['PRR'] < 1:
            assert row['ROR'] <= row['PRR'] + 0.01, \
                f"ROR ({row['ROR']}) should be <= PRR ({row['PRR']}) when PRR < 1"
