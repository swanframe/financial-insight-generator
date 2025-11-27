# Financial Insight Generator (FIG)

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Interface](https://img.shields.io/badge/interface-CLI-green.svg)
![LLM](https://img.shields.io/badge/LLM-optional-orange.svg)

Financial Insight Generator (FIG) is a modular Python toolkit that turns raw
transaction-level data (CSV/Excel) into:

- Clean, validated datasets  
- Useful financial KPIs and segments  
- Human-readable insight reports  
- An interactive CLI “assistant”, now optionally powered by an LLM

Think of it as a small, extensible **junior financial analyst** you can run
locally today and later plug into an LLM or web UI.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Directory Layout](#directory-layout)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Data](#data)
  - [Columns](#columns)
  - [Analytics](#analytics)
  - [Output](#output)
  - [UI & Language](#ui--language)
  - [LLM](#llm)
- [Usage](#usage)
  - [One-off report script](#one-off-report-script)
  - [CLI](#cli)
  - [LLM report modes](#llm-report-modes)
  - [Interactive assistant (chat)](#interactive-assistant-chat)
  - [Multilingual usage (EN / ID)](#multilingual-usage-en--id)
- [Quickstart](#quickstart)
- [Testing](#testing)
- [AI Engineering Highlights](#ai-engineering-highlights)
- [Next Steps](#next-steps)
- [License](#license)
- [Author](#author)

---

## Features

- **Data ingestion & cleaning**
  - Load CSV/Excel transaction data.
  - Validate required columns and types.
  - Normalize dates, filter invalid rows, and save a cleaned dataset.

- **Analytics & KPIs**
  - Overall metrics:
    - Revenue, cost, gross profit, gross margin.
    - Number of transactions, average order value (AOV).
    - Date range of the dataset.
  - Time series (e.g., revenue by month).
  - Segment metrics:
    - Category
    - Product
    - Customer
    - Channel
  - Trend analysis and a simple anomaly check on recent revenue.

- **Insight reports**
  - Template-based, deterministic report generation.
  - Optional **LLM-based narrative reports** that use the analytics output as structured context.
  - Three modes: `template`, `llm`, `hybrid`.

- **Interactive assistant (CLI)**
  - Rule-based commands for:
    - Summary / overview
    - Top categories / products / customers / channels
    - Trend and anomaly
    - Time series
  - **LLM-powered free-form questions** about the data when enabled.

- **Multilingual UX**
  - English and Bahasa Indonesia support.
  - The same language setting is respected across reports and chat.
  - LLM prompts are explicitly instructed to answer in the selected language.

---

## Tech Stack

- **Language:** Python 3.12
- **Core libraries:** `pandas`, `numpy`
- **CLI & tooling:** standard library (`argparse`, `pathlib`)
- **Testing:** `pytest`
- **LLM integration (optional):** `openai` Python SDK (or other providers via `fig.llm_client`)
- **i18n:** simple YAML-based localization (`locales/en.yaml`, `locales/id.yaml`)

---

## Architecture Overview

At a high level:

1. **Configuration** (`config.yaml` + `fig.config`)
2. **Data pipeline**
   - `fig.data_loader` (load CSV/Excel)
   - `fig.validation` (validate shapes/types)
   - `fig.preprocessing` (clean/normalize and optionally save cleaned CSV)
3. **Analytics** (`fig.analytics`)
   - Computes KPI dictionaries and DataFrames.
   - Main output is a **`metrics_bundle`**:
     ```python
     metrics_bundle = {
         "overall": {...},          # overall KPIs
         "time_series": df,         # revenue over time
         "segments": {...},         # category / product / customer / channel
         "monthly_trend": {...},    # MoM trend summary
         "anomaly": {...},          # recent anomaly info
     }
     ```
4. **Insight generation**
   - Template-based (`fig.insights.generate_full_report`).
   - LLM-based (`fig.llm_insights.generate_llm_report`) using:
     - Structured context from `metrics_bundle`
     - Prompts from `fig.llm_prompts`
     - Provider-agnostic calls via `fig.llm_client`
5. **CLI / Chat**
   - `fig.cli` for running the pipeline + report + interactive mode.
   - `fig.chatbot` for rule-based commands.
   - `fig.llm_chatbot` for free-form Q&A on top of the metrics.

Everything is wired so that **LLM features are optional**: when disabled, FIG acts as a traditional analytics + reporting tool.

---

## Directory Layout

Key files:

```text
.
├── config.yaml                 # Main configuration file
├── run_insights_report.py      # One-off report script (CLI)
├── src/
│   └── fig/
│       ├── config.py           # Config dataclasses + loader
│       ├── data_loader.py      # CSV/Excel loading
│       ├── validation.py       # Validation of raw transactions
│       ├── preprocessing.py    # Cleaning & normalization
│       ├── analytics.py        # Core metrics and metrics_bundle
│       ├── insights.py         # Template-based reports
│       ├── llm_client.py       # Provider-agnostic LLM client
│       ├── llm_prompts.py      # Prompt + context builders
│       ├── llm_insights.py     # LLM-based report generation
│       ├── chatbot.py          # Rule-based CLI assistant
│       ├── llm_chatbot.py      # LLM-powered free-form chat helper
│       ├── i18n.py             # Simple i18n wrapper
│       └── locales/
│           ├── en.yaml         # English strings
│           └── id.yaml         # Indonesian strings
└── tests/
    └── ...                     # Unit tests (analytics, config, LLM, etc.)
````

---

## Installation

1. **Clone the repo** and create a virtual environment.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional, for LLM features) Install a provider SDK. For OpenAI:

```bash
pip install "openai>=1.0.0"
```

4. The default `config.yaml` points to a sample CSV. Adjust paths as needed.

---

## Configuration

All configuration lives in `config.yaml` and is loaded via `fig.config.load_config`.

### Data

```yaml
data:
  input_path: "data/raw/sample_transactions.csv"
  date_format: "%Y-%m-%d"  # optional; let pandas infer if omitted
  parse_dates: true
```

### Columns

Logical → physical column mapping:

```yaml
columns:
  date: "order_date"       # required
  amount: "total_price"    # required
  cost: "cost"             # optional
  category: "category"     # optional
  product: "product_name"  # optional
  customer_id: "customer_id"
  channel: "sales_channel"
```

### Analytics

```yaml
analytics:
  time_granularity: "month"       # "day", "week", or "month"
  top_n: 5
  anomaly_lookback_days: 30
  anomaly_sigma_threshold: 2.0
```

### Output

```yaml
output:
  save_clean_data: true
  clean_data_path: "data/processed/cleaned_transactions.csv"

  save_metrics: true
  metrics_path: "data/processed/metrics.json"

  save_report: true
  report_path: "reports/financial_insights.txt"
```

### UI & Language

```yaml
ui:
  language: "en"   # or "id" for Bahasa Indonesia
```

This language is used for:

* Template-based reports
* CLI banners and messages
* Chatbot prompts and responses
* LLM prompts (the model is explicitly told which language to answer in)

You can also override it via CLI with `--lang`.

### LLM

The LLM section is **optional**. If omitted, FIG runs in pure template mode with `llm.enabled = false`.

Example configuration:

```yaml
llm:
  enabled: false          # master toggle; false = no LLM calls

  provider: "openai"      # logical provider name
  model: "gpt-4.1-mini"   # model name for your provider

  temperature: 0.3
  max_tokens: 800

  api_key_env_var: "OPENAI_API_KEY"
  timeout_seconds: 30

  # Report / assistant mode:
  # - "template": built-in templates only
  # - "llm":      LLM-only narrative reports
  # - "hybrid":   template report + LLM refinement
  mode: "template"

  # Max characters of structured context passed into prompts
  max_context_chars: 12000
```

> ⚠️ **Never put API keys in `config.yaml`.**
> Use environment variables instead:
>
> ```bash
> export OPENAI_API_KEY="sk-...your-key-here..."
> ```

---

## Usage

### One-off report script

Run the full pipeline + report generation:

```bash
python run_insights_report.py
```

Options:

```bash
python run_insights_report.py \
  --config config.yaml \
  --lang en \
  --report-mode template   # or llm / hybrid
```

* `--lang` overrides `ui.language` from config.
* `--report-mode` overrides `llm.mode` for this run.

### CLI

The CLI provides the same report plus optional interactive mode:

```bash
python -m fig.cli --config config.yaml
```

Flags:

```bash
python -m fig.cli \
  --config config.yaml \
  --lang en \
  --report-mode template \
  --interactive
```

* If `--interactive` is passed, the chat interface starts after the report.

### LLM report modes

The effective report mode is:

> CLI `--report-mode` (if provided) → `llm.mode` in config.yaml → `"template"`

* `template`
  Uses the original deterministic `generate_full_report(metrics_bundle, language)`.
  **No LLM calls** are made, even if `llm.enabled: true`.

* `llm`
  Uses `generate_llm_report(..., mode_override="llm")`:

  * Summarizes `metrics_bundle` into a compact, structured context.
  * Builds an LLM prompt via `fig.llm_prompts`.
  * Calls the provider via `fig.llm_client.generate_text`.

* `hybrid`
  Similar to `llm`, but the existing template-based report is passed into the prompt as a “draft” that the model can refine and expand—while still being constrained by the data.

If the LLM is disabled or misconfigured (missing key, unknown provider, etc.):

* The code **never crashes the pipeline**.
* A short note is added at the top:

  * EN:
    `"[Note] The LLM-based report could not be generated... Falling back to the template-based report."`
  * ID:
    `"[Catatan] Laporan LLM tidak dapat dibuat... Sistem kembali ke laporan berbasis template."`
* The underlying template-based report is still printed.

### Interactive assistant (chat)

Start the interactive assistant:

```bash
python -m fig.cli --config config.yaml --interactive
```

You can then type commands like:

* `summary`
* `top categories`
* `top products`
* `top customers`
* `top channels`
* `trend`
* `anomaly`
* `time series`
* `help`
* `exit`

These are handled by the **rule-based** logic in `fig.chatbot` and will behave the same whether or not the LLM is enabled.

#### Free-form questions (LLM-enhanced)

If `llm.enabled: true`, any input that does **not** match a known command is treated as a free-form question and routed through `fig.llm_chatbot`, for example:

* `Is there anything unusual about recent sales?`
* `Why did revenue increase in March?`
* `Which customers are driving most of the profit?`

Under the hood:

1. `metrics_bundle` is summarized via `fig.llm_prompts.summarize_metrics_bundle`.
2. A chat-style system prompt is built with `build_system_prompt_for_chat(language)`.
3. A user prompt is built with `build_user_prompt_for_question(...)`, including:

   * The structured metrics summary.
   * A short description of available CLI commands.
   * The user’s question.
4. `fig.llm_client.generate_text` calls the LLM.

The model is instructed to:

* Use **only** the metrics provided.
* Avoid inventing transactions, dates, or customers.
* Explain limitations honestly if the data does not support a detailed answer.

On configuration/provider errors (e.g., missing key, network issue), the assistant responds with a friendly note instead of crashing and suggests using the rule-based commands.

### Multilingual usage (EN / ID)

The effective language is:

> CLI `--lang` → `ui.language` in config.yaml → `"en"`

Examples:

* English report and chat:

  ```bash
  python -m fig.cli --config config.yaml --lang en
  ```

* Indonesian report and chat:

  ```bash
  python -m fig.cli --config config.yaml --lang id
  ```

For Indonesian:

* Template reports use translated headings and phrases from `locales/id.yaml`.
* Chat messages (`help`, `welcome`, `unknown_command`, etc.) are in Bahasa Indonesia.
* LLM system/user prompts explicitly ask the model to respond in Bahasa Indonesia.

---

## Quickstart

Fast path for trying the project locally:

```bash
# 1) Create and activate a virtualenv (example)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a one-off template-based report
python run_insights_report.py

# 4) Start the interactive assistant (template mode)
python -m fig.cli --config config.yaml --interactive
```

Enable LLM features (optional):

```bash
# 5) Set your API key (example for OpenAI)
export OPENAI_API_KEY="sk-...your-key-here..."

# 6) In config.yaml:
# llm:
#   enabled: true
#   provider: "openai"
#   model: "gpt-4.1-mini"
#   ...

# 7) Run with LLM-powered reports and chat
python -m fig.cli --config config.yaml --interactive --report-mode llm
```

---

## Testing

Run the full test suite:

```bash
pytest
```

The tests cover:

* Data & analytics core.
* Configuration loading (with and without `llm` section, invalid modes, etc.).
* Prompt builders (`fig.llm_prompts`).
* LLM client configuration behavior (`fig.llm_client`), including:

  * Missing API keys.
  * Unsupported providers.
* LLM report wrapper (`fig.llm_insights`):

  * Disabled LLM fallback.
  * Template mode skipping LLM calls.
  * Hybrid mode error fallback.
* LLM chat helper (`fig.llm_chatbot`), with mocked LLM responses.

All LLM-related tests run **fully offline**: they mock the client and never hit external APIs.

---

## AI Engineering Highlights

This repo is designed to demonstrate **AI Engineer** skills:

* **LLM-ready architecture**

  * Clear separation between data pipeline, analytics, and language generation.
  * `metrics_bundle` as a structured context that can be reused across UIs / backends.

* **Provider-agnostic LLM client**

  * `fig.llm_client` takes a `Config` object and hides provider details.
  * Swapping providers is mostly a matter of extending this single module and adjusting `config.yaml`.

* **Prompt & context design**

  * Prompts explicitly constrain the model to the known metrics and guard against hallucinations.
  * Context truncation (`max_context_chars`) keeps prompts bounded and predictable.

* **Multilingual UX**

  * Shared language selection logic for CLI, reports, and LLM prompts.
  * Explicit system instructions for English and Bahasa Indonesia.

* **Testability & reliability**

  * LLM usage is easy to mock and test.
  * Misconfiguration or provider errors never break the core pipeline; the system always falls back to a safe, deterministic path.

---

## Next Steps

If you want to extend FIG further, natural follow-ups include:

* Exposing the analytics + LLM layer as a **FastAPI** or **Django** web service.
* Adding a simple **React dashboard** that calls an API endpoint for:

  * Metrics & charts.
  * LLM-generated summaries.
  * Chat about your data.
* Adding support for more LLM providers by extending `fig.llm_client`.

The core pieces—structured analytics, i18n-aware prompts, and provider-agnostic LLM integration—are already in place.

---

## License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

## Author

**Rahman**

* GitHub: [https://github.com/swanframe](https://github.com/swanframe)
* LinkedIn: [https://www.linkedin.com/in/rahman-080902337](https://www.linkedin.com/in/rahman-080902337)