# Building a Data Analysis Application — A Guide for Domain Experts

*This guide is for researchers, clinicians, pharmacologists, and other domain experts who have data and methodology but not software engineering expertise. Claude Code will use this guide to walk you through building a complete, deployable data analysis application.*

---

## What You Need to Bring

You do not need to know how to code. You need three things:

### 1. Your Data Source

Tell Claude Code about your data. Any of these work:

- **A public dataset URL** (government data portal, research repository, API endpoint)
- **A file you have** (CSV, Excel, JSON, XML — any tabular format)
- **A database you can access** (provide connection details)
- **A description of data you need downloaded** (e.g., "FDA FAERS quarterly data from 2020-2024")

Key questions Claude Code will ask you:
- What does each table/file represent?
- How are tables related to each other? (What field links them?)
- What is one row? (One patient? One event? One transaction?)
- Are there duplicates that need handling?
- What fields have coded values? (e.g., "1=Male, 2=Female")
- How large is the dataset? (Rows, time span)

### 2. Your Methodology

Tell Claude Code how you want to analyze the data. Any of these work:

- **A published paper** — paste a DOI, URL, or describe the methods section
- **A textbook method** — name the statistical technique (e.g., "disproportionality analysis," "Kaplan-Meier survival curves," "logistic regression")
- **A formula** — write it out in plain language or math notation
- **An existing tool you want to replicate** — describe what it does and what is missing
- **A question you want answered** — "I want to know which drugs are associated with liver injury more than expected by chance"

Key questions Claude Code will ask you:
- What is the primary outcome or measure you care about?
- What statistical thresholds define a "significant" result in your field?
- What are known confounders or biases in this type of analysis?
- Should results be stratified by any variable? (age, sex, time period)
- What does a "positive result" look like? Can you give an example from a published study?
- What limitations or caveats must be shown to the end user?

### 3. Your Audience

Tell Claude Code who will use the finished application:

- **Just you** — a local script that generates a report is fine
- **Your research group** — a shared tool with some documentation
- **The public** — a web application anyone can access without installing anything
- **A specific community** — clinicians, patients, regulators, journalists

This determines whether we build a script, a notebook, or a deployed web application.

---

## What Claude Code Will Build

Based on your inputs, Claude Code follows a standard architecture pattern:

```
Your Data (any format)
    ↓
[ETL Script] — Download, clean, convert to Parquet
    ↓
Parquet Files (on HuggingFace, S3, or local)
    ↓
[DuckDB] — Fast SQL queries, no server needed
    ↓
[Analysis Engine] — Your methodology implemented in Python/SQL
    ↓
[Web UI] — Streamlit app (optional, for public deployment)
    ↓
[AI Report] — LLM generates narrative from results (optional)
```

### Components that get built:

| Component | What it does | When it is needed |
|-----------|-------------|-------------------|
| `data_download.py` | Fetches and converts your raw data to Parquet | Always |
| `analysis.py` | Implements your methodology as a batch script | Always |
| `app.py` | Web interface for interactive queries | If audience is beyond just you |
| `Dockerfile` | Container config for cloud deployment | If deploying to HuggingFace Spaces |
| `requirements.txt` | Python dependencies | Always |
| AI report prompt | System prompt tailored to your domain | If you want LLM-generated narrative reports |

---

## Step-by-Step Process

### Step 1: Describe Your Project

Start by telling Claude Code about your project in plain language. For example:

> "I study drug-induced liver injury. I have access to the FDA FAERS database and the LiverTox database from NIH. I want to build a tool where a user can type a drug name and see whether it has a statistically elevated liver injury signal compared to background, using the methodology from [this paper]. The tool should be free and accessible to hepatologists who don't code."

Or:

> "I have a CSV of 50,000 patient records with demographics, diagnoses, and lab values. I want to identify clusters of patients with similar lab trajectories using k-means clustering, then visualize the clusters. Only my research team will use this."

Or:

> "I want to replicate Table 3 from this paper [URL] but with updated data from 2020-2024 instead of 2015-2019. The methodology is Cox proportional hazards regression."

### Step 2: Data Ingestion

Claude Code will:
1. Write a script to download/load your data
2. Convert it to Parquet format (columnar, fast, compact)
3. Handle data quality issues (missing values, duplicates, format inconsistencies)
4. Optionally upload to HuggingFace Datasets for free cloud hosting

**Your role:** Verify the data looks right. Claude Code will show you row counts, sample rows, and data quality summaries. Flag anything that looks wrong.

### Step 3: Implement Your Methodology

Claude Code will:
1. Translate your methodology into SQL queries and Python code
2. Follow the 4-round analysis pattern:
   - **Round 1** — Data profiling (understand what we have)
   - **Round 2** — Frequency and distribution analysis
   - **Round 3** — Your specific statistical method (PRR, ROR, Cox regression, clustering, etc.)
   - **Round 4** — Summary and interpretation
3. Generate validation output you can check against known results

**Your role:** Validate the results. If you have a published paper with known results, check that the tool reproduces them. For example: "The paper reports PRR=2.3 for tetrabenazine + depression. Does our tool show approximately the same?"

### Step 4: Build the Interface (if needed)

For web deployment, Claude Code will build a Streamlit application with:
- Search/input fields appropriate to your use case
- Results tables with proper formatting
- Charts and visualizations
- CSV download for further analysis
- Proper disclaimers and limitation statements
- Optional: AI-powered report generation (bring your own API key)

**Your role:** Test the interface. Is the output understandable to your target audience? Are the right fields shown? Is anything confusing or missing?

### Step 5: Deploy (if needed)

For public access, Claude Code will deploy to HuggingFace Spaces:
- Free hosting, no server management
- Accessible via URL from any browser
- No installation required for end users

**Your role:** Share the URL with your community and gather feedback.

---

## Common Patterns by Domain

These are examples of analyses that fit this architecture well:

### Pharmacovigilance / Drug Safety
- **Data:** FDA FAERS, EudraVigilance, WHO VigiBase
- **Methods:** Disproportionality analysis (PRR, ROR, EBGM), Evans' criteria, time-to-onset analysis
- **Output:** Drug safety signal reports, drug class comparisons

### Epidemiology
- **Data:** CDC WONDER, WHO GHO, national health surveys
- **Methods:** Standardized rates, risk ratios, trend analysis, joinpoint regression
- **Output:** Disease burden dashboards, temporal trend reports

### Clinical Research
- **Data:** Clinical trial registries (ClinicalTrials.gov), patient registries
- **Methods:** Survival analysis, meta-analysis, forest plots, funnel plots
- **Output:** Evidence synthesis tools, trial landscape maps

### Public Health Surveillance
- **Data:** Syndromic surveillance feeds, lab reporting, environmental monitoring
- **Methods:** CUSUM, EWMA, Poisson regression, spatial clustering
- **Output:** Outbreak detection dashboards, environmental exposure maps

### Health Economics
- **Data:** Claims databases, cost databases, utilization data
- **Methods:** Cost-effectiveness analysis, budget impact modeling, DRG analysis
- **Output:** Cost comparison tools, resource utilization dashboards

---

## What You Do NOT Need to Know

- **Programming languages** — Claude Code writes all the Python, SQL, HTML/CSS
- **Cloud infrastructure** — HuggingFace Spaces handles hosting for free
- **Database administration** — DuckDB runs in-process, no server to manage
- **Web development** — Streamlit generates the UI from Python code
- **Docker/containers** — Claude Code writes the Dockerfile
- **Git/version control** — Claude Code handles commits and uploads

## What You DO Need to Know

- **Your domain** — what the data means, what constitutes a valid analysis, what the limitations are
- **Your methodology** — the statistical approach, even if described informally
- **Your audience** — who will use this and what decisions they will make with it
- **What "correct" looks like** — a published result, a known benchmark, or expert judgment to validate against

---

## Getting Started

Open Claude Code and say something like:

> "I want to build a [type of analysis] tool using [data source]. The methodology is [describe or link to paper]. The users will be [audience]. Can you help me build this following the pattern in lake_house_skill.md?"

Claude Code will ask you the questions listed above, then start building.
