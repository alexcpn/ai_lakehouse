"""
AI Data Lakehouse in a Box - DuckDB + Parquet Analysis

Demonstrates the concept from ai-datalakehouse.md:
Query NYC Yellow Taxi trip data directly from a local Parquet file
using DuckDB - no Kubernetes, no Trino, no Hive Metastore needed.
"""

import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PARQUET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yellow_tripdata_2022-01.parquet'))
OUTPUT_IMAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fare_analysis.png'))


def get_connection():
    conn = duckdb.connect(database=':memory:')
    conn.execute(f"CREATE VIEW taxi AS SELECT * FROM read_parquet('{PARQUET_PATH}')")
    return conn


def round1_schema_and_profiling(conn):
    print("\n--- Schema Discovery ---")
    schema = conn.execute("DESCRIBE SELECT * FROM taxi").fetchdf()
    print(schema.to_string(index=False))

    print("\n--- Data Profiling ---")
    profile = conn.execute("""
        SELECT
            COUNT(*) as total_rows,
            ROUND(AVG(fare_amount), 2) as avg_fare,
            ROUND(STDDEV(fare_amount), 2) as std_fare,
            ROUND(MIN(fare_amount), 2) as min_fare,
            ROUND(MAX(fare_amount), 2) as max_fare,
            ROUND(MEDIAN(fare_amount), 2) as median_fare
        FROM taxi
    """).fetchdf()
    print(profile.to_string(index=False))

    print("\n--- Data Quality ---")
    quality = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE fare_amount <= 0) as negative_or_zero_fares,
            COUNT(*) FILTER (WHERE fare_amount > 500) as extreme_fares,
            COUNT(*) FILTER (WHERE trip_distance <= 0) as zero_distance,
            COUNT(*) FILTER (WHERE trip_distance > 100) as extreme_distance
        FROM taxi
    """).fetchdf()
    print(quality.to_string(index=False))

    return {
        'total_rows': int(profile['total_rows'].iloc[0]),
        'avg_fare': float(profile['avg_fare'].iloc[0]),
        'median_fare': float(profile['median_fare'].iloc[0]),
        'negative_fares': int(quality['negative_or_zero_fares'].iloc[0]),
        'extreme_fares': int(quality['extreme_fares'].iloc[0]),
    }


def round2_correlation_analysis(conn):
    print("\n--- Distance Buckets vs Fare ---")
    distance_df = conn.execute("""
        SELECT
            CASE
                WHEN trip_distance BETWEEN 0 AND 5 THEN '0-5mi'
                WHEN trip_distance BETWEEN 5 AND 10 THEN '5-10mi'
                WHEN trip_distance BETWEEN 10 AND 20 THEN '10-20mi'
                WHEN trip_distance BETWEEN 20 AND 40 THEN '20-40mi'
                ELSE '40+mi'
            END as distance_bucket,
            COUNT(*) as trips,
            ROUND(AVG(fare_amount), 2) as avg_fare,
            ROUND(STDDEV(fare_amount), 2) as std_fare
        FROM taxi
        WHERE fare_amount > 0 AND fare_amount < 500
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print(distance_df.to_string(index=False))

    print("\n--- Time of Day Effect ---")
    hourly_df = conn.execute("""
        SELECT
            EXTRACT(HOUR FROM tpep_pickup_datetime) as pickup_hour,
            COUNT(*) as trips,
            ROUND(AVG(fare_amount), 2) as avg_fare
        FROM taxi
        WHERE fare_amount > 0 AND fare_amount < 500
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print(hourly_df.to_string(index=False))

    print("\n--- Day of Week Effect ---")
    daily_df = conn.execute("""
        SELECT
            DAYNAME(tpep_pickup_datetime) as day_name,
            EXTRACT(DOW FROM tpep_pickup_datetime) as dow,
            ROUND(AVG(fare_amount), 2) as avg_fare,
            ROUND(AVG(tip_amount), 2) as avg_tip,
            ROUND(AVG(trip_distance), 2) as avg_distance
        FROM taxi
        WHERE fare_amount > 0 AND fare_amount < 500
        GROUP BY 1, 2
        ORDER BY 2
    """).fetchdf()
    print(daily_df.to_string(index=False))

    return {
        'distance': distance_df,
        'hourly': hourly_df,
        'daily': daily_df,
    }


def round3_statistical_validation(conn, r2):
    print("\n--- Sampling 5% for Statistical Analysis ---")
    df = conn.execute("""
        SELECT
            trip_distance, fare_amount, tip_amount, passenger_count,
            EXTRACT(HOUR FROM tpep_pickup_datetime) as hour,
            EXTRACT(DOW FROM tpep_pickup_datetime) as dow
        FROM taxi
        WHERE fare_amount > 0 AND fare_amount < 500
          AND trip_distance > 0 AND trip_distance < 100
        USING SAMPLE 5 PERCENT (bernoulli)
    """).fetchdf()
    print(f"Sample size: {len(df)} rows")

    print("\n--- Correlation with fare_amount ---")
    corr = df[['fare_amount', 'trip_distance', 'tip_amount', 'passenger_count', 'hour']].corr()
    fare_corr = corr['fare_amount'].drop('fare_amount').sort_values(ascending=False)
    print(fare_corr.to_string())

    print("\n--- T-test: Weekend vs Weekday Fares ---")
    weekend = df[df['dow'].isin([0, 6])]['fare_amount']  # 0=Sunday, 6=Saturday
    weekday = df[~df['dow'].isin([0, 6])]['fare_amount']
    t_stat, p_value = stats.ttest_ind(weekend, weekday)
    print(f"Weekend mean: ${weekend.mean():.2f}  Weekday mean: ${weekday.mean():.2f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('NYC Yellow Taxi - January 2022 - Fare Analysis', fontsize=14)

    # 1. Distance vs Fare scatter
    ax = axes[0, 0]
    sample_plot = df.sample(min(5000, len(df)))
    ax.scatter(sample_plot['trip_distance'], sample_plot['fare_amount'], alpha=0.1, s=5)
    z = np.polyfit(sample_plot['trip_distance'], sample_plot['fare_amount'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, sample_plot['trip_distance'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    ax.set_xlabel('Trip Distance (mi)')
    ax.set_ylabel('Fare Amount ($)')
    ax.set_title('Trip Distance vs Fare')
    ax.set_ylim(0, 150)
    ax.legend()

    # 2. Avg fare by hour
    ax = axes[0, 1]
    hourly = r2['hourly']
    ax.bar(hourly['pickup_hour'], hourly['avg_fare'], color='steelblue')
    ax.set_xlabel('Pickup Hour')
    ax.set_ylabel('Avg Fare ($)')
    ax.set_title('Average Fare by Hour of Day')
    ax.set_xticks(range(0, 24, 3))

    # 3. Avg fare by day of week
    ax = axes[1, 0]
    daily = r2['daily'].sort_values('dow')
    ax.bar(daily['day_name'], daily['avg_fare'], color='coral')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Avg Fare ($)')
    ax.set_title('Average Fare by Day of Week')
    ax.tick_params(axis='x', rotation=45)

    # 4. Correlation coefficients
    ax = axes[1, 1]
    colors = ['green' if v > 0 else 'red' for v in fare_corr.values]
    ax.barh(fare_corr.index, fare_corr.values, color=colors)
    ax.set_xlabel('Correlation with Fare Amount')
    ax.set_title('Feature Correlations')
    ax.axvline(x=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {OUTPUT_IMAGE}")

    return {
        'sample_size': len(df),
        'fare_distance_corr': float(fare_corr['trip_distance']),
        'weekend_mean': float(weekend.mean()),
        'weekday_mean': float(weekday.mean()),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'fare_corr': fare_corr,
    }


def round4_synthesis(r1, r2, r3):
    peak_hour = r2['hourly'].loc[r2['hourly']['avg_fare'].idxmax()]
    low_hour = r2['hourly'].loc[r2['hourly']['avg_fare'].idxmin()]
    diff_pct = ((r3['weekend_mean'] - r3['weekday_mean']) / r3['weekday_mean']) * 100
    sig = "statistically significant" if r3['p_value'] < 0.001 else "not statistically significant"

    summary = f"""
FINDINGS SUMMARY
================
Dataset: {r1['total_rows']:,} NYC Yellow Taxi trips (January 2022)
Data Quality: {r1['negative_fares']:,} negative/zero fares, {r1['extreme_fares']:,} extreme fares (>$500)

1. FARE DRIVERS
   Trip distance is the dominant factor (r={r3['fare_distance_corr']:.2f}).
   Tip amount also correlates (r={float(r3['fare_corr']['tip_amount']):.2f}) — expected since tips scale with fare.
   Passenger count has negligible impact (r={float(r3['fare_corr']['passenger_count']):.2f}).

2. TIME OF DAY
   Peak avg fare at hour {int(peak_hour['pickup_hour'])}:00 (${float(peak_hour['avg_fare']):.2f}).
   Lowest avg fare at hour {int(low_hour['pickup_hour'])}:00 (${float(low_hour['avg_fare']):.2f}).

3. WEEKEND vs WEEKDAY
   Weekend mean: ${r3['weekend_mean']:.2f} | Weekday mean: ${r3['weekday_mean']:.2f} ({diff_pct:+.1f}%)
   T-test: t={r3['t_stat']:.3f}, p={r3['p_value']:.6f} — {sig}.

Median fare: ${r1['median_fare']:.2f} | Mean fare: ${r1['avg_fare']:.2f}
"""
    return summary


def main():
    print("=" * 60)
    print("AI Data Lakehouse in a Box — DuckDB + Parquet")
    print(f"Data: {PARQUET_PATH}")
    print("=" * 60)

    conn = get_connection()

    print("\n" + "=" * 60)
    print("ROUND 1: Schema Discovery and Data Profiling")
    print("=" * 60)
    r1 = round1_schema_and_profiling(conn)

    print("\n" + "=" * 60)
    print("ROUND 2: Correlation Analysis")
    print("=" * 60)
    r2 = round2_correlation_analysis(conn)

    print("\n" + "=" * 60)
    print("ROUND 3: Statistical Validation")
    print("=" * 60)
    r3 = round3_statistical_validation(conn, r2)

    print("\n" + "=" * 60)
    print("ROUND 4: Synthesis")
    print("=" * 60)
    summary = round4_synthesis(r1, r2, r3)
    print(summary)

    conn.close()


if __name__ == '__main__':
    main()
