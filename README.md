# Financial Insight Generator (FIG)

Financial Insight Generator (FIG) is a modular Python toolkit that turns raw
transaction-level data (CSV/Excel) into:

- Clean, validated datasets  
- Useful financial KPIs and segments  
- Human-readable insight reports and an interactive CLI “assistant”

Think of it as a small, extensible **junior financial analyst** you can run
locally today and plug into an LLM or web UI later.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration (`config.yaml`)](#configuration-configyaml)
- [Sample Dataset](#sample-dataset)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Roadmap & Future Enhancements](#roadmap--future-enhancements)
- [What This Demonstrates (AI Engineering Skills)](#what-this-demonstrates-ai-engineering-skills)
- [Tech Stack](#tech-stack)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Flexible data ingestion**
  - CSV support out of the box (Excel via `pandas.read_excel` if needed).
  - Designed for transaction-level data (one row = one order/transaction).
  - Configurable column mapping so different schemas map into a common
    internal model.

- **Data cleaning & validation**
  - Validates required fields (date, amount).
  - Checks for invalid dates and non-numeric amounts/costs.
  - Normalizes dates and numeric fields using `pandas`.
  - Produces validation + cleaning summaries.
  - Saves cleaned data to `data/processed/` (path is configurable).

- **Analytics & KPIs**
  - Overall metrics:
    - Total revenue, total cost, gross profit, gross margin.
    - Number of transactions, average order value (AOV).
  - Time-series metrics:
    - Revenue by day/week/month (configurable).
  - Segment metrics:
    - Revenue by category, product, customer, and channel.
    - Top-N segments by revenue (configurable).
  - Trend metrics:
    - Month-over-month revenue trend (last month vs previous).
  - Anomaly detection:
    - Last-day revenue anomaly check using a simple z-score vs recent history.

- **Natural-language insights**
  - Template-based text generation:
    - Overview summary.
    - Segment highlights (top categories/products/customers/channels).
    - Month-over-month trend explanation.
    - Daily anomaly summary.
  - A single `generate_full_report(...)` function assembles a complete report.

- **CLI / chat-like interface**
  - CLI entrypoint:
    - Runs the full pipeline: config → data → analytics → insights.
    - Prints a full text report.
    - Optional interactive shell with commands like:
      - `summary`
      - `top categories`
      - `top products`
      - `top customers`
      - `top channels`
      - `trend`
      - `anomaly`
      - `time series`

- **Clean, extensible architecture**
  - `src/fig` package with clear layers:
    - `config` → `data_loader` → `validation` → `preprocessing`
      → `analytics` → `insights` → `cli` / `chatbot`.
  - Easy to extend:
    - Add new KPIs or segment analyses in `analytics.py`.
    - Swap or augment template-based insights with LLMs later.

---

## Architecture Overview

The core flow:

```text
raw CSV/Excel
      ↓
   data_loader       (I/O only)
      ↓
   validation        (checks required columns, types)
      ↓
  preprocessing      (rename → normalize → clean)
      ↓
    analytics        (pure functions -> metrics & DataFrames)
      ↓
    insights         (template-based text)
      ↓
   cli/chatbot       (terminal UX, commands → insights)
````

A **metrics bundle** (a single Python dict) acts as the contract between
analytics, insights, and any future LLM/web integrations:

```python
metrics_bundle = {
    "overall": {...},
    "time_series": <DataFrame>,
    "segments": {
        "category": <DataFrame>,
        "product": <DataFrame>,
        "customer_id": <DataFrame>,
        "channel": <DataFrame>,
    },
    "monthly_trend": {...},
    "anomaly": {...},
}
```

Any consumer (CLI, notebook, API, LLM) can read from this bundle.

---

## Project Structure

```text
financial-insight-generator/
  README.md                      # Project documentation (this file)
  requirements.txt               # Python dependencies
  config.yaml                    # Config for input paths, mappings, analytics

  data/
    raw/
      sample_transactions.csv    # Synthetic sample dataset
    processed/
      cleaned_transactions.csv   # Cleaned output (generated)

  reports/                       # Optional: place for saving reports
    # (currently report is printed to console; can be saved here later)

  notebooks/
    exploration.ipynb            # Optional: EDA / experimentation

  src/
    fig/
      __init__.py                # Package init
      config.py                  # YAML loading & typed Config dataclasses
      data_loader.py             # CSV/Excel loading
      validation.py              # Basic schema / type checks, validation report
      preprocessing.py           # Column mapping, normalization, cleaning
      analytics.py               # KPIs, segments, trends, anomaly detection
      insights.py                # Natural-language text generation
      chatbot.py                 # Interactive terminal “assistant”
      cli.py                     # CLI entrypoint (`python -m fig.cli`)

  tests/
    __init__.py
    conftest.py                  # Shared fixtures (config, pipeline, metrics)
    test_config.py               # Config loading tests
    test_data_loader.py          # Data loading tests
    test_analytics.py            # Core analytics tests
    test_insights.py             # Report generation tests

  run_data_pipeline.py           # Script: run pipeline & print validation/cleaning summary
  run_analytics_demo.py          # Script: run analytics & print metrics summary
  run_insights_report.py         # Script: generate full text report
```

---

## Installation

### Requirements

* Python **3.10+** recommended
* macOS / Linux (Windows should also work with minor tweaks)
* No GPU required (pure CPU / pandas / numpy)

### Setup

```bash
# Clone your repo (example)
git clone https://github.com/<your-username>/financial-insight-generator.git
cd financial-insight-generator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Because the package uses a `src/` layout, ensure `src` is on your `PYTHONPATH`
when running scripts/tests directly:

```bash
export PYTHONPATH=src
```

You can put that line in your shell profile if you want it permanent.

---

## Configuration (`config.yaml`)

The configuration file controls:

* Input path
* Column mapping
* Analytics options
* Output paths

Example (included by default):

```yaml
data:
  input_path: "data/raw/sample_transactions.csv"
  date_format: "%Y-%m-%d"
  parse_dates: true

columns:
  # Required logical fields
  date: "order_date"
  amount: "total_price"

  # Optional fields
  cost: "cost"
  category: "category"
  product: "product_name"
  customer_id: "customer_id"
  channel: "sales_channel"

analytics:
  time_granularity: "month"        # "day", "week", or "month"
  top_n: 5                         # top N segments for categories/products/customers/channels
  anomaly_lookback_days: 30
  anomaly_sigma_threshold: 2.0

output:
  save_clean_data: true
  clean_data_path: "data/processed/cleaned_transactions.csv"

  # Reserved for future use (metrics/report saving)
  save_metrics: true
  metrics_path: "data/processed/metrics.json"

  save_report: true
  report_path: "reports/financial_insights.txt"
```

### Column mapping

Your CSV column names are mapped to a fixed internal schema:

* `date` → transaction date (required)
* `amount` → revenue / total order value (required)
* `cost` → cost of goods (optional)
* `category` → product/category label (optional)
* `product` → product name or ID (optional)
* `customer_id` → customer identifier (optional)
* `channel` → sales channel (optional)

Update `config.yaml` so each logical field points to the corresponding column
name in your data. If a field doesn’t exist in your dataset, remove or comment it.

---

## Sample Dataset

A synthetic sample dataset is included at:

* `data/raw/sample_transactions.csv`

It covers:

* Several months of orders
* Multiple categories, products, customers, and channels
* Reasonable prices and costs

This lets you run the full pipeline out of the box.

---

## Usage

Make sure your virtual environment is active and `PYTHONPATH=src`.

### 1. Run the data pipeline only

```bash
python run_data_pipeline.py
```

This will:

* Load `config.yaml`
* Load raw CSV (`data/raw/sample_transactions.csv`)
* Validate and clean data
* Print validation + cleaning summaries
* Save cleaned data to `data/processed/cleaned_transactions.csv`

### 2. Run analytics demo

```bash
python run_analytics_demo.py
```

This will:

* Run the data pipeline
* Compute overall metrics, time series, segment metrics, trend, anomaly
* Print a structured summary to the console

### 3. Generate a full insight report

```bash
python run_insights_report.py
```

This will:

* Run the data pipeline
* Build a `metrics_bundle`
* Generate a multi-section text report
* Print it to the console

(You can easily extend this script to also save the report to
`config.output.report_path`.)

### 4. Use the CLI / interactive assistant

The main CLI is provided by `fig.cli`:

```bash
# Print full report (non-interactive)
python -m fig.cli --config config.yaml

# Print full report and start interactive mode
python -m fig.cli --config config.yaml --interactive

# Using defaults (config.yaml in project root)
python -m fig.cli -i
```

In interactive mode, you can use commands like:

```text
fig> help
fig> summary
fig> top categories
fig> top products
fig> top customers
fig> top channels
fig> trend
fig> anomaly
fig> time series
fig> exit
```

---

## Running Tests

From the project root, with your venv active:

```bash
export PYTHONPATH=src
pytest
```

The tests cover:

* Config loading and basic validation.
* Data loading from the sample CSV.
* Core analytics (overall metrics, segment revenue).
* Insight generation (full report contains expected sections).

---

## Roadmap & Future Enhancements

Planned / easy next steps:

* **Metrics & report persistence**

  * Save `metrics_bundle` to JSON.
  * Save full text report to `reports/financial_insights.txt` using
    `config.output.report_path`.

* **LLM integration**

  * Use `metrics_bundle` as a structured context for an LLM (e.g. OpenAI GPT).
  * Let users ask free-form questions about their data:

    * “Why did revenue drop last month?”
    * “Which customers are growing fastest?”
  * Keep `insights.py` as a fallback when LLMs are unavailable.

* **Web/API layer**

  * Wrap the pipeline in a FastAPI / Flask service.
  * Expose endpoints for:

    * Uploading CSVs
    * Triggering analysis
    * Fetching metrics and reports as JSON/text

* **Automation / scheduling**

  * Run `run_insights_report.py` daily via cron / CI.
  * Push reports to Slack, email, or S3.

* **Packaging & Docker**

  * Add `pyproject.toml` for packaging.
  * Add a simple Dockerfile for containerized deployment.

---

## What This Demonstrates (AI Engineering Skills)

* **Data engineering**

  * Config-driven ingestion pipeline.
  * Schema normalization and validation.
  * Clean separation between raw and processed data.

* **Analytics & ML engineering patterns**

  * Pure, testable analytics functions (`analytics.py`).
  * Basic anomaly detection using statistics (z-score).
  * Segmentation and trend analysis via `pandas`.

* **AI system design**

  * Clear boundary between:

    * Data layer (loading & preprocessing),
    * Analytics layer (metrics),
    * Insight layer (natural-language generation),
    * Interface layer (CLI/chat).
  * Metrics bundle designed as a structured “contract” that an LLM or
    other consumers could use later.

* **Production-minded Python project structure**

  * `src/` layout, tests, and config-based behavior.
  * Ready to be extended with packaging, Docker, or deployment scripts.

> **Note:** This project is intended as an educational / analytics tool.
> It is not financial advice and should not be used as the sole basis for
> real-world financial decisions.

---

## Tech Stack

* **Language**: Python 3.10+
* **Data / analytics**: `pandas`, `numpy`
* **Config**: `pyyaml`
* **Testing**: `pytest`
* **Interface**: Standard Python CLI (no external frameworks)

---

## License

This project is licensed under the **MIT License**.
You can modify, distribute, and use it in commercial or non-commercial projects, subject to the usual MIT conditions.

---

## Contact

If you have questions, suggestions, or want to discuss the project:

* GitHub: [@your-username](https://github.com/your-username)
* Email: [your.email@example.com](mailto:your.email@example.com)
* LinkedIn: [https://www.linkedin.com/in/your-profile/](https://www.linkedin.com/in/your-profile/)

Happy analyzing 🚀