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
- (Optional) a vector index for similarity search over transactions

Think of it as a small, extensible **junior financial analyst** you can run
locally today and later plug into an LLM, vector database, or web UI.

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
  - [Embeddings](#embeddings)
  - [Vector Store](#vector-store)
- [Usage](#usage)
  - [One-off report script](#one-off-report-script)
  - [CLI](#cli)
  - [LLM report modes](#llm-report-modes)
  - [Interactive assistant (chat)](#interactive-assistant-chat)
  - [Multilingual usage (EN / ID)](#multilingual-usage-en--id)
  - [Vector index build & search](#vector-index-build--search)
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
  - Segment metrics by:
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

- **Vector search over transactions (optional, RAG-friendly)**
  - Builds embeddings for each cleaned transaction and indexes them in a vector store.
  - Supports local-first providers (persistent Chroma or an in-memory store).
  - Exposes a simple CLI to search for transactions similar to a natural-language query, ready to be reused in future RAG pipelines.

---

## Tech Stack

- **Language:** Python 3.12
- **Core libraries:** `pandas`, `numpy`
- **CLI & tooling:** standard library (`argparse`, `pathlib`)
- **Testing:** `pytest`
- **LLM integration (optional):** `openai` Python SDK (or other providers via `fig.llm_client`)
- **Embeddings & vector DB (optional):**
  - Provider-agnostic embeddings (`openai` or a local `dummy` provider)
  - `chromadb` for persistent vector storage
- **i18n:** YAML-based localization (`locales/en.yaml`, `locales/id.yaml`)

---

## Architecture Overview

At a high level:

1. **Configuration** (`config.yaml` + `fig.config`)
2. **Data pipeline**
   - `fig.data_loader` – load CSV/Excel
   - `fig.validation` – validate shapes/types
   - `fig.preprocessing` – clean/normalize and optionally save cleaned CSV
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
   - `fig.cli` for running the pipeline + report + optional interactive mode.
   - `fig.chatbot` for rule-based commands.
   - `fig.llm_chatbot` for free-form Q&A on top of the metrics.
6. **Embeddings & vector store**
   - `fig.embeddings.embed_texts(texts, config)`:
     - Reads `config.embeddings` and produces text embeddings via a chosen provider.
   - `fig.vector_store`:
     - `build_or_update_index(df, config, rebuild=False)` builds/updates a transaction-level vector index.
     - `query_similar_transactions(query_text, config, top_k=None)` returns the most similar transactions for a given natural-language query.

Both **LLM** and **vector** functionality are fully optional: when disabled, FIG behaves as a traditional analytics + reporting tool.

---

## Directory Layout

Key files:

```text
.
├── config.yaml                 # Main configuration file
├── run_insights_report.py      # One-off report script (CLI)
├── run_build_vector_index.py   # Build/update the transaction vector index
├── run_vector_search.py        # Search the transaction vector index from the CLI
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
│       ├── embeddings.py       # Shared embeddings helper (openai / dummy)
│       ├── vector_store.py     # Vector DB abstraction (build & query)
│       ├── i18n.py             # Simple i18n wrapper
│       └── locales/
│           ├── en.yaml         # English strings
│           └── id.yaml         # Indonesian strings
└── tests/
    └── ...                     # Unit tests (analytics, config, LLM, vector, etc.)
````

---

## Installation

1. **Clone the repo** and create a virtual environment.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional, for LLM or real embeddings) Set up a provider. For OpenAI, just export an API key:

```bash
export OPENAI_API_KEY="sk-...your-key-here..."
```

> The `openai` and `chromadb` packages are already included in `requirements.txt`.

4. Make sure Python can find the `src/` package. Either:

* Set `PYTHONPATH` once in your shell:

  ```bash
  export PYTHONPATH=src
  ```

  or
* Prefix commands with `PYTHONPATH=src`, e.g.:

  ```bash
  PYTHONPATH=src python run_insights_report.py
  ```

The examples below assume `src` is on `PYTHONPATH` (either via `export` or by prefixing commands).

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

### Embeddings

Embeddings configuration is shared by the vector store and any future RAG-style components:

```yaml
embeddings:
  provider: "openai"              # "openai" for real embeddings, or "dummy" for a local demo
  model: "text-embedding-3-small" # ignored for dummy, used for real providers
  api_key_env_var: "OPENAI_API_KEY"
  timeout_seconds: 30
```

Notes:

* If the `embeddings` section is omitted, FIG derives sensible defaults from the `llm` section:

  * `provider` ← `llm.provider`
  * `api_key_env_var` ← `llm.api_key_env_var`
  * `timeout_seconds` ← `llm.timeout_seconds`
* `provider: "dummy"` uses a deterministic, local-only embedding implementation:

  * No API key and no network required.
  * Great for offline demos or when you’re out of quota.

### Vector Store

The vector store section controls whether vector features are enabled and how the index is stored:

```yaml
vector_store:
  enabled: false                  # master toggle for all vector features

  provider: "chroma"              # "chroma" (persistent) or "in_memory"

  persist_path: "data/vector_store"
  collection_name: "fig_transactions"

  default_top_k: 5
```

Notes:

* When `enabled: false`, scripts that depend on vector search print a friendly note and exit; the rest of FIG is unaffected.
* `provider: "chroma"`:

  * Uses `chromadb` as a local-first vector database.
  * Persists the index under `persist_path`.
* `provider: "in_memory"`:

  * Uses an in-process Python implementation.
  * Does not write anything to disk (handy for tests and simple demos).

---

## Usage

> **Note:** In the examples below, assume `src` is on `PYTHONPATH` (e.g. `export PYTHONPATH=src`).

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
  Uses `generate_full_report(metrics_bundle, language)`.
  **No LLM calls** are made, even if `llm.enabled: true`.

* `llm`
  Uses `generate_llm_report(..., mode_override="llm")`:

  * Summarizes `metrics_bundle` into a compact, structured context.
  * Builds an LLM prompt via `fig.llm_prompts`.
  * Calls the provider via `fig.llm_client.generate_text`.

* `hybrid`
  Similar to `llm`, but the template-based report is passed into the prompt as a “draft” that the model can refine and expand, while still being constrained by the data.

If the LLM is disabled or misconfigured (missing key, unknown provider, etc.):

* The code **never crashes the pipeline**.
* A short note is added at the top:

  * EN: `"[Note] The LLM-based report could not be generated... Falling back to the template-based report."`
  * ID: `"[Catatan] Laporan LLM tidak dapat dibuat... Sistem kembali ke laporan berbasis template."`
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

These are handled by the **rule-based** logic in `fig.chatbot` and behave the same whether or not the LLM is enabled.

#### Free-form questions (LLM-enhanced)

If `llm.enabled: true`, any input that does **not** match a known command is treated as a free-form question and routed through `fig.llm_chatbot`, for example:

* `Is there anything unusual about recent sales?`
* `Why did revenue increase in March?`
* `Which customers are driving most of the profit?`

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
* LLM prompts explicitly ask the model to respond in Bahasa Indonesia.

### Vector index build & search

Once vector features are configured, you can build and query a transaction index from the CLI.

**Build or update the index:**

```bash
python run_build_vector_index.py --config config.yaml
```

Optional:

```bash
python run_build_vector_index.py --config config.yaml --rebuild
```

> If `vector_store.enabled: false`, this script prints a short note and exits without affecting anything else.

**Search for similar transactions:**

```bash
python run_vector_search.py \
  --config config.yaml \
  --query "large electronics orders in March"
```

Optional flags:

```bash
python run_vector_search.py \
  --config config.yaml \
  --query "high value grocery purchases" \
  --top-k 3 \
  --lang en
```

The search CLI prints the top-k similar transactions, including date, category, product, amount, and a similarity score.

---

## Quickstart

Fast path for trying the project locally:

```bash
# 1) Create and activate a virtualenv (example)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Make sure src/ is on PYTHONPATH
export PYTHONPATH=src

# 4) Run a one-off template-based report
python run_insights_report.py

# 5) Start the interactive assistant (template mode)
python -m fig.cli --config config.yaml --interactive
```

Enable LLM features (optional):

```bash
# 6) Set your API key (example for OpenAI)
export OPENAI_API_KEY="sk-...your-key-here..."

# 7) In config.yaml:
# llm:
#   enabled: true
#   provider: "openai"
#   model: "gpt-4.1-mini"
#   ...

# 8) Run with LLM-powered reports and chat
python -m fig.cli --config config.yaml --interactive --report-mode llm
```

Vector search (optional, RAG-friendly):

```bash
# 9) In config.yaml, enable the vector store:
# vector_store:
#   enabled: true
#   provider: "chroma"
#   persist_path: "data/vector_store"
#   collection_name: "fig_transactions"

# 10) Build the index
python run_build_vector_index.py --config config.yaml

# 11) Search similar transactions
python run_vector_search.py \
  --config config.yaml \
  --query "large electronics orders in March"
```

Offline / dummy vector demo (no API keys needed):

```bash
# Example overrides in config.yaml:
# embeddings:
#   provider: "dummy"
#   model: "ignored-for-dummy"
#   api_key_env_var: "IGNORED_FOR_DUMMY"
#
# vector_store:
#   enabled: true
#   provider: "in_memory"
#   persist_path: "data/vector_store"
#   collection_name: "fig_transactions"

# Build the in-memory index using dummy embeddings
python run_build_vector_index.py --config config.yaml

# Run a local-only search
python run_vector_search.py \
  --config config.yaml \
  --query "large electronics orders in March"
```

---

## Testing

Run the full test suite (assuming `src` is on `PYTHONPATH`):

```bash
PYTHONPATH=src pytest
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
* Embeddings and vector store:

  * Config wiring for `embeddings` and `vector_store`.
  * In-memory vector index build & query.
  * Vector search CLI behavior with mocked backends.

All LLM/embedding/vector-related tests run **fully offline**: they mock the client or use the `dummy` provider and never hit external APIs.

---

## AI Engineering Highlights

This repo is designed to demonstrate **AI Engineering** skills:

* **LLM-ready architecture**

  * Clear separation between data pipeline, analytics, and language generation.
  * `metrics_bundle` as a structured context that can be reused across UIs / backends.

* **Provider-agnostic LLM client**

  * `fig.llm_client` takes a `Config` object and hides provider details.
  * Swapping providers is mostly a matter of extending this single module and adjusting `config.yaml`.

* **Config-driven embeddings & vector store**

  * `fig.embeddings` and `fig.vector_store` use dedicated config sections with sensible defaults derived from the LLM config.
  * Embeddings provider can be a real API (`openai`) or a local `dummy` implementation, and the vector store can be persistent (`chroma`) or in-memory.

* **Prompt & context design**

  * Prompts explicitly constrain the model to the known metrics and guard against hallucinations.
  * Context truncation (`max_context_chars`) keeps prompts bounded and predictable.

* **Multilingual UX**

  * Shared language selection logic for CLI, reports, and LLM prompts.
  * Explicit system instructions for English and Bahasa Indonesia.

* **Testability & reliability**

  * LLM and vector usage is easy to mock and test.
  * Misconfiguration or provider errors never break the core pipeline; the system always falls back to a safe, deterministic path.

* **Vector- / RAG-ready**

  * Transaction-level vectors and a simple similarity API (`query_similar_transactions`) make it straightforward to build RAG-style flows that retrieve relevant transactions before calling an LLM.

---

## Next Steps

If you want to extend FIG further, natural follow-ups include:

* Using the vector store + embeddings to build a **RAG layer**:

  * Retrieve similar transactions and key metrics.
  * Feed both into an LLM to answer “why” questions grounded in actual data.
* Exposing the analytics + LLM layer as a **FastAPI** or **Django** web service.
* Adding a simple **React dashboard** that calls an API endpoint for:

  * Metrics & charts.
  * LLM-generated summaries.
  * Chat about your data.
* Adding support for more LLM/embedding providers by extending `fig.llm_client` and `fig.embeddings`.

The core pieces—structured analytics, i18n-aware prompts, provider-agnostic LLM client, and a vector-ready index—are already in place.

---

## License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

## Author

**Rahman**

* GitHub: [https://github.com/swanframe](https://github.com/swanframe)
* LinkedIn: [https://www.linkedin.com/in/rahman-080902337](https://www.linkedin.com/in/rahman-080902337)