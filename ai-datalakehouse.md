# From Data Lakehouse to AI Data Lakehouse

*How an AI coding agent built a complete FDA drug safety analysis tool from a single conversation — and what that means for the Data Lakehouse.*

**By Alex Punnen**

---

## Introduction

Three years ago, I wrote about the journey [from Data Warehouse to Data Lake to Data Lakehouse](https://medium.com/@alexpunnen/from-data-warehouse-to-data-lake-to-data-lakehouse-7f6c8c1b5e3a) — how S3 storage, columnar formats like Parquet, and SQL engines like Trino/Presto gave us a cost-effective way to do analytics at scale. The architecture was sound. The bottleneck was the human.

To get value from a Data Lakehouse, you still needed someone who could write complex SQL, someone who understood statistical methods, and — if you wanted anything beyond aggregations — someone trained in scikit-learn, pandas, feature engineering, and the full ML toolkit. The data was democratized. The skills to extract insight from it were not.

That has changed. Agentic AI — LLMs that can reason, write code, execute it, observe results, and iterate — has collapsed the skill barrier. To make this concrete: I used [Claude Code](https://claude.ai/code) (Anthropic's AI coding agent) to build a complete FDA drug safety analysis tool — from raw government data to a live web application with publication-quality statistical output. The kind of work that previously required a team of data engineers, statisticians, and web developers, done in a single conversation.

This is what I call the **AI Data Lakehouse** — not a new product, but a new paradigm. The same architecture we already built, now with an AI agent as the primary query interface instead of a human analyst. And more importantly: an AI agent that can *build the entire analytical application* from a domain expert's description.

> The best tool is the one that removes the need for the previous tool. — A pattern in technology

---

## Quick Recap: What We Built Before

For those who haven't read the earlier article, here is the architecture in brief:

**Data Lake** = S3/Object Storage. Cheap, durable, unlimited. You dump data here — Parquet, JSON, CSV, whatever. No schema enforcement, no SQL interface. Just storage.

**Data Lakehouse** = Data Lake + SQL Metadata Layer + Query Engine. We used:
- **S3** (or MinIO on bare metal) for storage
- **Apache Hive Metastore** (backed by Postgres) for table metadata and schema
- **Trino/Presto** as the distributed SQL query engine

This gave us the ability to load Parquet files into S3, create SQL table definitions pointing to them, and run analytical queries across millions of rows in seconds — all at a fraction of the cost of a traditional data warehouse.

The key insight from that article: **Don't treat a Data Lakehouse as a database.** No efficient updates, no ACID transactions (practically). It is optimized for append-only analytical workloads — log data, event data, counter data, historical records. Load it once, query it many times.

That constraint is still true. But it turns out, it maps perfectly to what AI agents are good at.

---

## The Old Way: Getting Value from Data Required a Stack of Skills

Consider what it took to answer a non-trivial business question from a Data Lakehouse before AI agents:

**Question:** *"What factors most influence taxi fare amounts in NYC, and can we predict fares for new trips?"*

Using the NYC Taxi dataset from the earlier article (2.4 million rows in Parquet on S3), here's what the traditional approach looked like:

### Step 1: SQL Exploration (Analyst)

```sql
SELECT COUNT(*) FROM nyc_in_parquet.tlc_yellow_trip_2022;
-- 2,463,931 rows

SELECT AVG(fare_amount), STDDEV(fare_amount),
       PERCENTILE_APPROX(fare_amount, 0.5) as median
FROM nyc_in_parquet.tlc_yellow_trip_2022
WHERE fare_amount > 0 AND fare_amount < 500;
```

### Step 2: Feature Engineering (Data Scientist, Python)

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('trino://trino:8080/hive/nyc_in_parquet')
df = pd.read_sql('SELECT * FROM tlc_yellow_trip_2022', engine)

# Feature engineering - requires domain knowledge
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()

# Clean outliers - requires understanding of the data
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]
df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
```

### Step 3: Model Selection and Training (Data Scientist, scikit-learn)

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

features = ['trip_distance', 'hour', 'day_of_week', 'is_weekend',
            'passenger_count', 'PULocationID', 'DOLocationID']

X = df[features].dropna()
y = df.loc[X.index, 'fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Try multiple models - requires ML knowledge
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
print(f"R² Score: {r2_score(y_test, rf.predict(X_test)):.3f}")

# Feature importance - requires interpretation skills
for name, imp in sorted(zip(features, rf.feature_importances_),
                         key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")
```

### Step 4: Visualization and Reporting (Analyst again)

Build dashboards in Redash, write a summary, present to stakeholders.

**Total skills required:** SQL, Python, pandas, scikit-learn, statistics, domain knowledge, visualization tools. **Total people involved:** Usually 2-3. **Time:** Days to weeks.

---

## The New Way: AI Agent + Data Lakehouse

Now consider the same question with an AI agent that has:
1. Access to the Trino SQL endpoint (via tool/function calling or MCP)
2. Ability to execute Python code (for statistical analysis and plotting)
3. The table schema as context

### What the interaction looks like

**User:** "What factors most influence taxi fare amounts? Show me the key drivers and any interesting patterns."

**What the AI agent does autonomously:**

**Round 1 — Schema Discovery and Data Profiling**
```sql
-- Agent explores the schema
SHOW COLUMNS FROM nyc_in_parquet.tlc_yellow_trip_2022;

-- Agent profiles the data
SELECT COUNT(*) as total_rows,
       AVG(fare_amount) as avg_fare,
       STDDEV(fare_amount) as std_fare,
       MIN(fare_amount) as min_fare,
       MAX(fare_amount) as max_fare,
       APPROX_PERCENTILE(fare_amount, 0.5) as median_fare
FROM nyc_in_parquet.tlc_yellow_trip_2022;

-- Agent checks for data quality issues
SELECT COUNT(*) as negative_fares
FROM nyc_in_parquet.tlc_yellow_trip_2022
WHERE fare_amount <= 0;
```

**Round 2 — Correlation Analysis (Agent decides this is needed)**
```sql
-- Distance vs fare
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
FROM nyc_in_parquet.tlc_yellow_trip_2022
WHERE fare_amount > 0 AND fare_amount < 500
GROUP BY 1 ORDER BY 1;

-- Time-of-day effect
SELECT HOUR(tpep_pickup_datetime) as pickup_hour,
       COUNT(*) as trips,
       ROUND(AVG(fare_amount), 2) as avg_fare
FROM nyc_in_parquet.tlc_yellow_trip_2022
WHERE fare_amount > 0 AND fare_amount < 500
GROUP BY 1 ORDER BY 1;

-- Day-of-week effect (we did this manually before!)
SELECT date_format(tpep_pickup_datetime, '%a') as day,
       ROUND(AVG(fare_amount), 2) as avg_fare,
       ROUND(AVG(tip_amount), 2) as avg_tip,
       ROUND(AVG(trip_distance), 2) as avg_distance
FROM nyc_in_parquet.tlc_yellow_trip_2022
WHERE fare_amount > 0 AND fare_amount < 500
GROUP BY 1;
```

**Round 3 — Statistical Validation (Agent writes and executes Python)**
```python
# Agent generates this code, executes it, reads the output
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Agent pulls a sample for statistical testing
# (it knows not to pull 2.4M rows into memory unnecessarily)
query = """
SELECT trip_distance, fare_amount, tip_amount, passenger_count,
       HOUR(tpep_pickup_datetime) as hour,
       DOW(tpep_pickup_datetime) as dow
FROM nyc_in_parquet.tlc_yellow_trip_2022
WHERE fare_amount > 0 AND fare_amount < 500
  AND trip_distance > 0 AND trip_distance < 100
TABLESAMPLE BERNOULLI (5)
"""
df = pd.read_sql(query, engine)

# Correlation matrix
corr = df.corr()['fare_amount'].sort_values(ascending=False)
print("Correlation with fare_amount:")
print(corr)

# Statistical test: weekend vs weekday fares
weekend = df[df['dow'].isin([6, 7])]['fare_amount']
weekday = df[~df['dow'].isin([6, 7])]['fare_amount']
t_stat, p_value = stats.ttest_ind(weekend, weekday)
print(f"\nWeekend vs Weekday fare t-test: t={t_stat:.3f}, p={p_value:.6f}")

# Generate visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... agent generates the full plotting code
plt.savefig('fare_analysis.png')
```

**Round 4 — Agent synthesizes findings into natural language:**

> "Trip distance is the dominant factor influencing fare (r=0.87), followed by time of day — late night rides (11PM–5AM) average 18% higher fares, likely due to surcharges. Weekend fares are statistically higher than weekday fares (p<0.001) but the effect size is small (~$0.80). Passenger count has negligible impact. The tip percentage is highest for short trips (under 3 miles) — possibly a floor effect where riders tip a minimum regardless of fare."

**Total skills required from the user:** The ability to ask a clear question.
**Total people involved:** 1.
**Time:** Minutes.

---

## What Changed — and What Did Not

Let's be precise about what the AI agent replaced and what it did not.

### What's Obsolete for 80% of Business Analytics

| Old Skill | Replaced By |
|-----------|-------------|
| Writing complex SQL by hand | Agent generates and iterates on SQL |
| pandas/DataFrame wrangling | Agent writes Python when SQL isn't enough |
| Feature engineering for tabular data | Agent discovers features through exploration |
| Model selection (RF vs XGBoost vs...) | Agent picks the right statistical test or model |
| scikit-learn pipeline construction | Agent generates the pipeline code |
| Matplotlib/Seaborn visualization | Agent creates plots as part of analysis |
| Interpreting statistical results | Agent explains in business language |

This is the "80%" that I believe is being automated away. The analyst who spent three days writing SQL, cleaning data in pandas, trying three models, and building a dashboard — that workflow collapses to a conversation.

### What's Not Obsolete

**1. Data Modeling — More Important Than Ever**

This is the critical insight: **the quality of the AI agent's analysis is bounded by the quality of your data model.**

If your Lakehouse has a clean star schema with well-named columns, proper data types, and logical table relationships, the agent will produce excellent analysis. If your data is dumped into S3 as raw JSON blobs with cryptic field names, the agent will struggle just like a human analyst would — except it will hallucinate more confidently.

The Data Lakehouse architecture from my earlier article — Hive Metastore providing schema, Parquet providing columnar efficiency, proper table definitions — is not just an optimization anymore. It is the **foundation that makes AI analysis possible.**

```sql
-- This schema is self-documenting. An AI agent can reason about it.
CREATE TABLE nyc_in_parquet.tlc_yellow_trip_2022 (
    vendorid INTEGER,
    tpep_pickup_datetime TIMESTAMP,
    tpep_dropoff_datetime TIMESTAMP,
    passenger_count DOUBLE,
    trip_distance DOUBLE,
    fare_amount DOUBLE,
    tip_amount DOUBLE,
    pulocationid INTEGER,    -- Pickup location zone
    dolocationid INTEGER,    -- Dropoff location zone
    payment_type INTEGER     -- 1=Credit, 2=Cash, 3=No charge...
)
WITH (FORMAT = 'PARQUET',
    EXTERNAL_LOCATION = 's3a://test/warehouse/nyc_in_parquet.db/tlc_yellow_trip_2022');
```

Good column names, proper types, documented enums — these become the interface between your data and the AI. **Data modeling is the new feature engineering.**

**2. Domain Expertise for Validation**

The agent told us weekend fares are higher. Is that because of surcharges, demand, or longer trips? The agent can hypothesize, but a domain expert needs to validate. An agent might find a correlation between pickup location and fare that is actually just reflecting airport surcharges — obvious to a domain expert, invisible to the model.

The AI does not know what it does not know. This is where hallucination risk is highest in analytical workflows.

**3. Real-time ML Systems**

Fraud detection, recommendation engines, real-time pricing — these still need traditional ML pipelines. An agentic AI answering questions is not the same as a model serving predictions at 10,000 requests per second. The Data Lakehouse was never the right place for these workloads anyway.

**4. Unstructured Data Pipelines**

Computer vision, NLP on raw text, audio processing — these require specialized models and training pipelines. Though even here, foundation models are eating into the need for custom training.

---

## The AI Data Lakehouse Architecture

Here is what the architecture looks like now. Note how little has changed in the infrastructure layer — we are adding an AI agent on top of the same stack.

```
┌─────────────────────────────────────────────────┐
│                  User / Analyst                  │
│          (asks questions in English)             │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              AI Agent (LLM + Tools)              │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │SQL Tool  │  │Python    │  │Visualization  │  │
│  │(Trino)   │  │Executor  │  │Tool           │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  Capabilities:                                   │
│  - Schema discovery (SHOW TABLES, DESCRIBE)      │
│  - Query generation and execution                │
│  - Statistical analysis (scipy, statsmodels)     │
│  - Iterative refinement based on results         │
│  - Natural language explanation                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│            Trino / Presto SQL Engine             │
│         (Distributed Query Processing)           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│         Hive Metastore (Schema Layer)            │
│    Table definitions, column metadata, stats     │
│              (backed by Postgres)                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           S3 / Object Storage                    │
│    Parquet files, partitioned by date/key        │
│         (MinIO on bare metal / AWS S3)           │
└─────────────────────────────────────────────────┘
```

The infrastructure is identical to what we built before. The only addition is the AI agent layer, which connects to Trino via JDBC/REST and has a Python execution environment.

### Practical Implementation: MCP + Claude/OpenAI

The most practical way to implement this today is via the **Model Context Protocol (MCP)** or simple function calling:

```python
# Simplified: AI Agent with Trino SQL tool
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "execute_sql",
        "description": "Execute a SQL query against the Trino Data Lakehouse. "
                       "Available schemas: nyc_in_parquet (NYC taxi data), "
                       "sales_data (company sales), web_logs (clickstream).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute against Trino"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "execute_python",
        "description": "Execute Python code for statistical analysis or visualization. "
                       "Available libraries: pandas, scipy, matplotlib, numpy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
]

# The agent loop - ask question, let AI iterate
messages = [
    {"role": "user", "content": "Analyze the NYC taxi data. "
     "What are the key fare drivers and any surprising patterns?"}
]

# Agent iterates autonomously - executing SQL, analyzing results,
# running Python, generating charts - until it has a complete answer
while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        # Execute the tool call (SQL or Python)
        tool_result = execute_tool(response.content)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_result})
    else:
        # Agent is done - print final analysis
        print(response.content[0].text)
        break
```

An MCP server wrapping Trino makes this even cleaner — the agent discovers available tables, schemas, and even column statistics through the protocol, rather than needing them hardcoded.

---

## Why OLAP/Lakehouse and Not Just "Give the Agent a Database"?

You might ask — why not just point the AI agent at a Postgres database? Why does the Lakehouse architecture matter?

Three reasons:

### 1. Cost at Scale

The NYC taxi dataset we used is 2.4 million rows in one month. A year is ~30 million rows. A decade is 300 million. An enterprise with clickstream data, IoT sensors, or transaction logs can easily hit billions of rows.

A Postgres instance serving this data would cost thousands per month in compute alone — running 24/7 whether anyone is querying or not. The Lakehouse? The data sits in S3 at pennies per GB. Trino spins up workers only when queries run. In a serverless setup (AWS Athena, which is Presto under the hood), you pay literally per query.

An AI agent that runs 50 queries to answer one question costs the same as a human running 50 queries. But the storage cost for the data it analyzes is orders of magnitude lower in a Lakehouse.

### 2. Schema-on-Read Flexibility

Databases enforce schema-on-write. If someone added a new field to the taxi data, you need an ALTER TABLE and migration. In the Lakehouse, the Parquet file simply has the new column. Update the Hive table definition and the AI agent can immediately query it.

This flexibility matters when AI agents are exploring data. They work best with wide tables — many columns to discover patterns in. Lakehouse architectures make it cheap to store wide, denormalized data because Parquet's columnar format means unused columns have zero query cost.

### 3. Separation of Compute and Storage

This is the architectural property that makes AI agents economically viable as the primary query interface.

An AI agent exploring data generates more queries than a human. It profiles every column, runs correlation checks, tests hypotheses, generates multiple views. In a traditional database, this compute load competes with production workloads. In a Lakehouse, the agent's Trino workers are isolated — they can scale up, run 50 queries in parallel, and scale back down without affecting anything else.

---

## The New Data Team

If this paradigm holds — and I believe the evidence is already strong — the composition of data teams changes:

**Before (Traditional Data Lakehouse):**
- Data Engineers — build pipelines, maintain infrastructure
- Data Analysts — write SQL, build dashboards
- Data Scientists — build ML models, feature engineering
- ML Engineers — productionize models

**After (AI Data Lakehouse):**
- Data Engineers — build pipelines, maintain infrastructure, **design schemas for AI consumption**
- Domain Experts — ask questions, validate AI-generated insights, guide analysis
- AI/Platform Engineers — maintain the agent infrastructure, fine-tune prompts, build guardrails

The Data Analyst and Data Scientist roles don't disappear overnight. But the entry-level tasks — the SQL monkey work, the "run a random forest on this CSV" tasks, the "make me a dashboard showing X" requests — those are automated first. What remains is the judgment: knowing which questions to ask, validating that answers make business sense, and recognizing when the AI is confidently wrong.

---

## What You Need to Get Started

If you already have a Data Lakehouse (or even just data in S3 with a query engine), you are closer than you think:

### 1. Well-Modeled Data (You probably need to improve this)
- Clean column names (not `col_1`, `col_2` but `pickup_datetime`, `fare_amount`)
- Proper data types (not everything as VARCHAR)
- Table and column descriptions in the Hive Metastore
- Document enum values (payment_type: 1=Credit, 2=Cash, etc.)

### 2. SQL Endpoint (You probably already have this)
- Trino, Presto, Athena, Spark SQL, DuckDB, ClickHouse — any OLAP engine with a SQL interface
- Network-accessible from wherever your agent runs

### 3. AI Agent with Tool Use (New component)
- Claude, GPT-4, or any model with function calling
- MCP server or custom tool wrapper for your SQL engine
- Optional: Python execution sandbox for statistical analysis
- Optional: Visualization tool integration

### 4. Guardrails (Critical for production)
- Read-only SQL access (the agent should never INSERT/UPDATE/DELETE)
- Query timeout limits (prevent runaway full-table scans)
- Cost limits per session
- Human-in-the-loop for high-stakes decisions

---

## A Note on DuckDB — The Local AI Data Lakehouse

One development worth highlighting: **DuckDB** has emerged as a compelling option for the AI Data Lakehouse pattern, especially for small-to-medium datasets.

DuckDB is an in-process OLAP database (think SQLite but for analytics) that can directly query Parquet files — including Parquet files on S3. No Hive Metastore needed. No Trino cluster to manage.

```sql
-- DuckDB can query S3 Parquet files directly
SELECT AVG(fare_amount), COUNT(*)
FROM read_parquet('s3://bucket/nyc_taxi/*.parquet')
WHERE trip_distance > 10;
```

For an AI agent doing exploratory analysis, DuckDB eliminates the infrastructure overhead entirely. The agent can:
1. Download or reference Parquet files from S3
2. Query them directly with DuckDB (in-process, zero setup)
3. Run statistical analysis in the same Python process
4. Return results to the user

This is the "AI Data Lakehouse in a box" — and for datasets under a few hundred GB, it may be all you need.

---

## Case Study: FDA Drug Safety Signal Detection — Built Entirely with Claude Code

The taxi fare analysis demonstrates the concept. But to prove the AI Data Lakehouse pattern works for problems that actually matter, I built something genuinely useful: a **free, zero-install pharmacovigilance signal detection tool** for FDA adverse event data.

Here is the important part: **I did not write the code.** I described the problem, pointed to the data source, referenced the statistical methodology, and Claude Code built the entire application — ETL pipeline, statistical engine, web interface, cloud deployment. A tool that would have taken a team of engineers and statisticians weeks to build.

### The Problem

The FDA collects voluntary reports of drug side effects through its [FAERS system](https://www.fda.gov/drugs/fdas-adverse-event-reporting-system-faers) (FDA Adverse Event Reporting System). Researchers use this data to detect safety signals — statistical associations between drugs and adverse events that might indicate previously unknown risks.

Here is what exists today:
- The **FDA's public dashboard** shows raw report counts. "Valbenazine has 47 depression reports." But is 47 a lot? Compared to what? A blockbuster drug will have more reports for *everything*. Raw counts are misleading without statistical context.
- **Published FAERS studies** do the proper analysis — disproportionality statistics like PRR and ROR that compare a drug's adverse event rate against the background. But these studies take weeks to months, require SAS or R expertise, and end up behind paywalls.

No free, zero-install tool existed for the analysis that pharmacovigilance actually requires.

### What I Told Claude Code

The conversation went roughly like this:

> "I want to build a pharmacovigilance signal detection tool using FDA FAERS data. The methodology is disproportionality analysis — PRR, ROR, chi-squared, Evans' criteria. Users should be able to search for a drug and see all its adverse event signals, or compare drugs in a class. Host it on HuggingFace Spaces so anyone can use it without installing anything. Here's a reference paper: Yokoi et al. 2023 on VMAT2 inhibitors and depression — tetrabenazine shows a signal, valbenazine does not. Use that as the validation benchmark."

That was the domain input. I did not specify DuckDB vs Postgres, did not choose Streamlit vs Flask, did not write any SQL. Claude Code made those architectural decisions and built:

### What Claude Code Built

**1. Data pipeline** (`faers_download.py`) — Downloads quarterly FAERS ZIP files from FDA, extracts the `$`-delimited ASCII files, converts them to Parquet via DuckDB, deduplicates cases, and uploads to HuggingFace Datasets.

**2. Statistical engine** (`faers_analysis.py`) — Implements the full disproportionality analysis: 2x2 contingency tables for every drug-AE pair, PRR, ROR, chi-squared, 95% confidence intervals via log-normal approximation, Evans' criteria signal flagging. Handles the real-world messiness of FAERS data — mixed age units (years/months/decades/days), non-standardized drug names, duplicate case reports.

**3. Web application** (`app.py`) — Streamlit interface with two modes: single drug safety profile and multi-drug class comparison. Bar charts, volcano plots, forest plots, downloadable CSV. Signal rows highlighted in red. Proper disclaimer about voluntary reporting limitations.

**4. AI report generator** — Optional feature where users provide their own Claude or GPT API key. The computed statistics are sent to the LLM, which generates a publication-quality safety narrative: background on the drug's pharmacology, clinical interpretation of the signals, confounder identification (e.g., antidepressants will always show a "depression" signal because they are prescribed *to depressed patients*), and limitations.

**5. Cloud deployment** — Dockerfile configured for HuggingFace Spaces, Parquet dataset uploaded to HuggingFace Hub. The app queries 2.9 million cases remotely via DuckDB's `hf://` protocol. Zero infrastructure to maintain.

### The Architecture

```
┌──────────────────────────────────────────────────┐
│  HuggingFace Dataset: fda-faers-parquet          │
│  7 Parquet tables, 2.9M cases (2023-2024)        │
│  DEMO, DRUG, REAC, OUTC, RPSR, THER, INDI       │
└────────────────────┬─────────────────────────────┘
                     │ hf:// protocol (DuckDB httpfs)
                     ▼
┌──────────────────────────────────────────────────┐
│  HuggingFace Space: Streamlit Web UI             │
│                                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │ Search: [drug name]          [Analyze]      │ │
│  │                                             │ │
│  │ → Statistical signals (PRR, ROR, Evans')    │ │
│  │ → Bar charts, volcano plots, forest plots   │ │
│  │ → CSV download                              │ │
│  │                                             │ │
│  │ [Generate AI Report]  ← BYOK Claude/GPT    │ │
│  │ → Publication-quality safety narrative       │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  DuckDB in-process → queries HF Dataset          │
│  scipy for CI/p-values                           │
│  LLM (Claude/GPT) for report generation          │
└──────────────────────────────────────────────────┘
```

The entire stack costs nothing to run. HuggingFace hosts the data and the application for free. DuckDB runs in-process — no database server. The only cost is the optional LLM API call for report generation, paid by the end user with their own key.

### What Would This Have Cost Before?

Let me be specific about what this tool replaces:

**A published FAERS disproportionality study** (like Yokoi et al. 2023, *Expert Opinion on Drug Safety*) typically involves:
- A pharmacovigilance researcher or biostatistician ($100K+/year salary)
- Weeks of data cleaning, SAS/R programming, and analysis
- Institutional access to statistical software licenses ($5K-$15K/year for SAS)
- Peer review cycle (months)
- Result: one paper, about one drug class, frozen in time at publication

**A commercial pharmacovigilance platform** (Oracle Argus, IQVIA, etc.) costs $50K-$500K/year.

**This tool** does the same core analysis — same PRR, same ROR, same Evans' criteria — for any drug, on demand, for free. A clinician wondering whether a patient's symptom might be drug-related can check in 30 seconds. A researcher can screen an entire drug class in minutes and export the data for a publication.

It is not a replacement for rigorous pharmacovigilance. It is a democratization of the first step — the signal detection scan that tells you where to look.

### The AI Layer: From Numbers to Narrative

The statistical analysis runs without any AI. This is critical — the tool is useful on its own. The PRR, ROR, chi-squared, confidence intervals, signal flags — all computed by DuckDB and scipy.

The AI adds a qualitatively different capability. When a user clicks "Generate AI Report," the LLM receives all the computed statistics and produces a structured safety report:

- **Background** on the drug's pharmacology and approved indications — contextualized from the LLM's training data
- **Key findings** with specific PRR values, confidence intervals, and case counts
- **Clinical context** — distinguishing known/labeled effects from potentially novel signals
- **Confounder identification** — flagging indication bias, notoriety bias, the Weber effect
- **Limitations** appropriate to voluntary reporting data

This is the part that no dashboard can do. A table showing "Drug X + Depression: PRR=2.3, p<0.001" is data. The AI saying "This signal is consistent with the dopamine-depleting mechanism of Drug X and aligns with the FDA label update from 2018, but the effect size is smaller than reported by [Author et al.]" is *interpretation*. That is what turns a data tool into something approaching a research assistant.

### Validation: VMAT2 Inhibitors and Depression

To verify the tool produces correct results, I used a known benchmark: Yokoi et al. 2023 studied VMAT2 inhibitors (tetrabenazine, deutetrabenazine, valbenazine) and found that tetrabenazine had a disproportionate depression signal while valbenazine did not.

The tool reproduces this finding. Tetrabenazine shows a depression signal (PRR > 2, meeting Evans' criteria). Valbenazine does not (PRR < 1). This took a research team months to publish. In our tool, it takes seconds.

### Try It

The tool is live at [alexcpn-faers-signal-detection.hf.space](https://alexcpn-faers-signal-detection.hf.space/). Source code is in this repository under `app/` and `python/`. The dataset covers 2.9 million unique cases from 2023Q1 to 2024Q4.

### What This Means for Domain Experts

The FAERS tool is an example, not an endpoint. The pattern generalizes:

1. **You have domain expertise** — you know the data, the methodology, the questions worth asking
2. **You describe the problem** to an AI coding agent (Claude Code, in this case)
3. **The agent builds the application** — ETL, statistics, web interface, deployment
4. **You validate and iterate** — check the results against known benchmarks, refine the UI

This inverts the traditional model. Instead of domain experts writing specifications for engineers to implement over weeks, the domain expert *is* the builder. The AI handles the engineering. The human provides the judgment that makes the tool trustworthy.

A pharmacologist who has never written Python can describe Evans' criteria and get a working signal detection tool. An epidemiologist can describe joinpoint regression and get a trend analysis dashboard. The bottleneck is no longer "can we build it?" — it is "do we know what to build?"

That is the real promise of the AI Data Lakehouse: not just AI querying data, but AI building the tools that query data, guided by the people who understand what the data means.

---

## Conclusion: The Lakehouse Was the Right Bet — But the Bigger Bet is on Domain Experts

Three years ago, building a Data Lakehouse felt like an infrastructure optimization — cheaper than a data warehouse, more structured than a raw data lake. The payoff was cost savings and flexibility.

What we didn't fully see was that the Lakehouse architecture — clean schemas, columnar storage, SQL interface, separated compute and storage — was also building the ideal substrate for AI-driven analysis. Every investment in data modeling, in proper Parquet partitioning, in clean table definitions, now compounds because it makes the AI agent more effective.

But the FAERS case study reveals something bigger than an architectural insight. The real shift is this: **the people closest to the problems can now build the tools to solve them.**

A pharmacovigilance researcher does not need a software team to build a signal detection tool. An epidemiologist does not need a data engineering department to build a disease surveillance dashboard. A clinician does not need to learn Python to create a drug interaction checker from public data. They need their domain expertise, a clear description of what they want, and an AI agent that can translate that description into working software.

The old ML toolkit — pandas, scikit-learn, feature engineering, model selection — was the bridge technology. It was what we needed when humans had to manually extract insights from data. With agentic AI, the bridge is no longer needed for the vast majority of analytical workloads. And crucially, the *engineers who built those bridges* are no longer the bottleneck for *domain experts who need to cross them*.

**The new stack is simple: store data well, describe your analysis clearly, and let the AI build the tool.**

The FAERS tool — 2.9 million cases, seven data tables, disproportionality analysis, interactive web UI, AI-generated reports, live deployment — was built in a single Claude Code conversation. Not because the engineering was trivial, but because the engineering is no longer the hard part. Knowing which drugs to compare, which statistical thresholds matter, which confounders to flag, what the results mean for a patient — *that* is the hard part. And that knowledge lives with domain experts, not with software engineers.

If you are a researcher, a clinician, or a domain expert sitting on data you wish you could analyze: you are closer than you think. You do not need to learn to code. You need to describe your problem clearly, point to your data, and let an AI agent do the rest.

---

*The FAERS pharmacovigilance tool is live at [alexcpn-faers-signal-detection.hf.space](https://alexcpn-faers-signal-detection.hf.space/) — no install required. Source code and a step-by-step guide for domain experts building similar tools are in this repository. For the full Kubernetes-based Lakehouse infrastructure, see [presto_in_kubernetes](https://github.com/alexcpn/presto_in_kubernete).*
